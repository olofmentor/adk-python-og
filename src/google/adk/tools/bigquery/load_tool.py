# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import uuid
from typing import Any, Optional

from google.api_core.exceptions import NotFound
from google.auth.credentials import Credentials
from google.cloud import bigquery

from . import client
from ..tool_context import ToolContext
from .config import BigQueryToolConfig, WriteMode


def _detect_source_format(file_path: str) -> bigquery.SourceFormat:
  lower = file_path.lower()
  if lower.endswith(".csv"):
    return bigquery.SourceFormat.CSV
  if lower.endswith(".json") or lower.endswith(".ndjson"):
    return bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
  if lower.endswith(".parquet"):
    return bigquery.SourceFormat.PARQUET
  if lower.endswith(".avro"):
    return bigquery.SourceFormat.AVRO
  # Default to CSV
  return bigquery.SourceFormat.CSV


def _table_exists(
    bq_client: bigquery.Client, project_id: str, dataset_id: str, table_id: str
) -> bool:
  try:
    bq_client.get_table(f"{project_id}.{dataset_id}.{table_id}")
    return True
  except NotFound:
    return False


def _get_table_schema_fields(
    bq_client: bigquery.Client, project_id: str, dataset_id: str, table_id: str
) -> list[bigquery.SchemaField]:
  table = bq_client.get_table(f"{project_id}.{dataset_id}.{table_id}")
  return list(table.schema or [])


def _quote_identifier(identifier: str) -> str:
  # Use backticks to quote as BigQuery standard SQL identifier
  return f"`{identifier}`"


async def load_file_to_bigquery(
    *,
    project_id: str,
    dataset_id: str,
    table_id: str,
    file_path: str,
    if_table_exists: str = "append",  # one of: append, replace, upsert
    unique_key_columns: Optional[list[str]] = None,
    csv_has_header: bool = True,
    credentials: Credentials,
    config: BigQueryToolConfig,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Load a local file into BigQuery with schema autodetection and optional upsert.

  Args:
    project_id: GCP project id.
    dataset_id: BigQuery dataset id.
    table_id: BigQuery table id.
    file_path: Local path to the input file (.csv, .json/.ndjson, .parquet, .avro).
    if_table_exists: Behavior when target table exists: 'append', 'replace', or 'upsert'.
    unique_key_columns: Required when if_table_exists == 'upsert'. List of column names forming a unique key to match target rows.
    csv_has_header: Applies when source is CSV. If True, the first row is treated as header.
    credentials: Injected by the framework. Google auth credentials.
    config: Injected tool configuration. Write mode and limits.
    tool_context: Tool execution context (unused except for parity and future use).

  Returns:
    A dict with status and details including created/existing table, rows processed, and effective schema.
  """
  try:
    if not config or config.write_mode == WriteMode.BLOCKED:
      return {
          "status": "ERROR",
          "error_details": "Write operations are blocked by BigQuery tool configuration.",
      }

    if config.write_mode == WriteMode.PROTECTED:
      return {
          "status": "ERROR",
          "error_details": (
              "Protected write mode only allows writes to anonymous session datasets. "
              "Use execute_sql in a session or set write_mode=ALLOWED."
          ),
      }

    if not os.path.exists(file_path):
      return {"status": "ERROR", "error_details": f"File not found: {file_path}"}

    bq_client = client.get_bigquery_client(project=project_id, credentials=credentials)

    source_format = _detect_source_format(file_path)
    destination = f"{project_id}.{dataset_id}.{table_id}"

    table_already_exists = _table_exists(bq_client, project_id, dataset_id, table_id)

    # Configure load job
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = source_format
    job_config.autodetect = True
    if source_format == bigquery.SourceFormat.CSV:
      job_config.skip_leading_rows = 1 if csv_has_header else 0
      job_config.field_delimiter = ","
      job_config.quote_character = '"'
      job_config.allow_quoted_newlines = True
    # Configure write disposition based on existence and requested action
    if table_already_exists:
      if if_table_exists.lower() == "append":
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
      elif if_table_exists.lower() == "replace":
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
      elif if_table_exists.lower() == "upsert":
        # Upsert handled via staging + MERGE; keep default (WRITE_EMPTY) for staging
        pass
      else:
        return {
            "status": "ERROR",
            "error_details": "Invalid if_table_exists. Use 'append', 'replace', or 'upsert'.",
        }
    else:
      # Create table on the fly if missing
      job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY

    if if_table_exists.lower() == "upsert" and not unique_key_columns:
      return {
          "status": "ERROR",
          "error_details": "unique_key_columns must be provided for upsert mode.",
      }

    if if_table_exists.lower() == "upsert" and table_already_exists:
      # Load into a staging table, then MERGE
      staging_table_id = f"{table_id}_staging_{uuid.uuid4().hex[:8]}"
      staging_fqtn = f"{project_id}.{dataset_id}.{staging_table_id}"

      staging_job_config = bigquery.LoadJobConfig()
      staging_job_config.source_format = source_format
      staging_job_config.autodetect = True
      if source_format == bigquery.SourceFormat.CSV:
        staging_job_config.skip_leading_rows = 1 if csv_has_header else 0
        staging_job_config.field_delimiter = ","
        staging_job_config.quote_character = '"'
        staging_job_config.allow_quoted_newlines = True

      with open(file_path, "rb") as f:
        load_job = bq_client.load_table_from_file(
            f, staging_fqtn, job_config=staging_job_config
        )
      load_result = load_job.result()

      # Build and run MERGE statement
      target_cols = [c.name for c in _get_table_schema_fields(bq_client, project_id, dataset_id, table_id)]
      staging_cols = [c.name for c in _get_table_schema_fields(bq_client, project_id, dataset_id, staging_table_id)]
      # Use the intersection of columns for update/insert
      join_keys = unique_key_columns or []
      common_non_key_cols = [
          c for c in target_cols if c in staging_cols and c not in set(join_keys)
      ]
      if not common_non_key_cols:
        # Still allow MERGE to only match without updates, but usually user expects updates
        pass

      on_clause = " AND ".join(
          [
              f"T.{_quote_identifier(k)} = S.{_quote_identifier(k)}"
              for k in join_keys
          ]
      )
      set_clause = ", ".join(
          [f"T.{_quote_identifier(c)} = S.{_quote_identifier(c)}" for c in common_non_key_cols]
      )
      insert_columns = [c for c in staging_cols]
      insert_values = [f"S.{_quote_identifier(c)}" for c in insert_columns]

      merge_sql_lines = [
          f"MERGE `{project_id}.{dataset_id}.{table_id}` T",
          f"USING `{project_id}.{dataset_id}.{staging_table_id}` S",
          f"ON {on_clause}",
      ]
      if set_clause:
        merge_sql_lines.append("WHEN MATCHED THEN UPDATE SET " + set_clause)
      merge_sql_lines.append(
          "WHEN NOT MATCHED THEN INSERT ("
          + ", ".join([_quote_identifier(c) for c in insert_columns])
          + ") VALUES ("
          + ", ".join(insert_values)
          + ")"
      )
      merge_sql = "\n".join(merge_sql_lines)

      query_job = bq_client.query(merge_sql, project=project_id)
      query_job.result()

      # Clean up staging table
      try:
        bq_client.delete_table(staging_fqtn, not_found_ok=True)
      except Exception:
        pass

      # Fetch final schema
      final_schema = [
          {"name": f.name, "type": f.field_type, "mode": f.mode}
          for f in _get_table_schema_fields(bq_client, project_id, dataset_id, table_id)
      ]
      return {
          "status": "SUCCESS",
          "action": "upsert",
          "table": destination,
          "rows": int(getattr(load_result, "output_rows", 0) or 0),
          "schema": final_schema,
      }

    # Non-upsert path: direct load to destination
    with open(file_path, "rb") as f:
      load_job = bq_client.load_table_from_file(
          f, destination, job_config=job_config
      )
    load_result = load_job.result()

    final_schema = [
        {"name": f.name, "type": f.field_type, "mode": f.mode}
        for f in _get_table_schema_fields(bq_client, project_id, dataset_id, table_id)
    ]

    return {
        "status": "SUCCESS",
        "action": (
            "create" if not table_already_exists else ("replace" if job_config.write_disposition == bigquery.WriteDisposition.WRITE_TRUNCATE else "append")
        ),
        "table": destination,
        "rows": int(getattr(load_result, "output_rows", 0) or 0),
        "schema": final_schema,
    }

  except Exception as ex:  # pylint: disable=broad-except
    return {
        "status": "ERROR",
        "error_details": str(ex),
    }