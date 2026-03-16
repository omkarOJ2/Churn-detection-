import boto3, os
from datetime import datetime, timezone, timedelta

def lambda_handler(event, context):
    region      = os.environ['AWS_REGION_NAME']
    rt_endpoint = os.environ['REALTIME_ENDPOINT']
    idle_hours  = int(os.environ.get('IDLE_HOURS', 1))

    sm = boto3.client('sagemaker',  region_name=region)
    cw = boto3.client('cloudwatch', region_name=region)

    # Check if realtime endpoint exists
    try:
        sm.describe_endpoint(EndpointName=rt_endpoint)
    except sm.exceptions.ClientError:
        print(f'Endpoint {rt_endpoint} not found — already deleted')
        return {'action': 'already_deleted', 'message': f'{rt_endpoint} not found'}

    # Check invocations in the last idle_hours window
    now   = datetime.now(timezone.utc)
    start = now - timedelta(hours=idle_hours)

    resp = cw.get_metric_statistics(
        Namespace='AWS/SageMaker',
        MetricName='Invocations',
        Dimensions=[
            {'Name': 'EndpointName', 'Value': rt_endpoint},
            {'Name': 'VariantName',  'Value': 'AllTraffic'}
        ],
        StartTime=start, EndTime=now,
        Period=3600 * idle_hours, 
        Statistics=['Sum']
    )

    invocations = sum(d['Sum'] for d in resp['Datapoints'])
    print(f'{rt_endpoint} — invocations in last {idle_hours}h: {invocations}')

    if invocations == 0:
        sm.delete_endpoint(EndpointName=rt_endpoint)
        print(f'Deleted {rt_endpoint} — no traffic in {idle_hours}h — billing stopped')
        return {
            'action': 'deleted',
            'invocations_1hr': 0,
            'message': f'{rt_endpoint} deleted — no traffic — billing stopped'
        }

    print(f'Kept {rt_endpoint} — {invocations} calls in last {idle_hours}h')
    return {
        'action': 'kept_alive',
        'invocations_1hr': invocations,
        'message': f'{rt_endpoint} ACTIVE ({invocations} calls) — keeping alive'
    }
