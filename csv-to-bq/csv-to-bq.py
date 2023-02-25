from google.cloud import bigquery #pip install google-cloud-bigquery==3.3.1

def csv_to_bq(request):
  client = bigquery.Client()
  job_config = bigquery.LoadJobConfig(
      #schema=[
      #    bigquery.SchemaField("EMPLOYEE_ID", "INTEGER"),
      #],
  skip_leading_rows=1)
  table_id = "warren-says.data.predictions"
  uri = "gs://raw-data_bucket/predict_poc.csv"
  load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)

  load_job.result()  # Waits for the job to complete.
  destination_table = client.get_table(table_id)
  print("Loaded {} rows.".format(destination_table.num_rows))
  return f'Success!'
