from google.cloud import bigquery #pip install google-cloud-bigquery==3.3.1

def csvtobq(request):
  content_type = request.headers['content-type']
  request_json = request.get_json(silent=True)

  if request_json and 'table_id' in request_json and 'uri' in request_json and 'replace_flag': #MMP 2023-03-11
      table_id = request_json['table_id']
      uri = request_json['uri']
      replace_flag = request_json['replace_flag'] #MMP 2023-03-11
  else:
      raise ValueError("JSON is invalid, or missing 'table_id' or 'uri' property")
      
  client = bigquery.Client()
  job_config = bigquery.LoadJobConfig(
      #schema=[
      #    bigquery.SchemaField("EMPLOYEE_ID", "INTEGER"),
      #],
  skip_leading_rows=1)
  #table_id = "warren-says.data.predictions"
  #uri = "gs://raw-data_bucket/predict_poc.csv"
  
  #+MMP 2023-03-11-
  if replace_flag == 1:
    job_config = bigquery.LoadJobConfig(
      write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
      source_format=bigquery.SourceFormat.CSV,
      skip_leading_rows=1,
    )
  #-MMP 2023-03-11
  
  load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)

  load_job.result()  # Waits for the job to complete.
  destination_table = client.get_table(table_id)
  print("Loaded {} rows.".format(destination_table.num_rows))
  return f'Success!'
