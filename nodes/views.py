import datetime
import json
import os.path
import pickle
import pandas as pd
import numpy as np
import keras
import requests
import csv
import logging
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core import serializers
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.conf import settings
from .models import Node
from django.http import HttpResponseBadRequest
from keras.models import load_model
from nodes.models import Nodes, Feeds, CropImage
from .forms import RegisterForm, ImageUploadForm, CSVImportForm

from PIL import Image  # Import the Image module from PIL (Python Imaging Library)
import io



# Create your views here.

def feeds_preprocess(node_id, lws, c_time):
    print("in preprocess")
    rec = Feeds.objects.filter(node_id=node_id).order_by('-id').values()
    print(rec)
    if not rec: 
        return {'duration': 0, 'event': 0}
    last_rec = rec[0]
    print(last_rec)
    timediff = c_time - last_rec['created_at']
    timediff = int((timediff.total_seconds()) / 3600)
    last_lws = last_rec['LWS']
    print(last_lws)
    if (last_lws >= 46000) and (lws < 46000):
        # event starting point
        # TODO : change in last param and add duration in current param
        # duration = timediff
        # event = 1
        return {'duration': timediff, 'event': 1}
    elif (last_lws < 46000) and (lws >= 46000):
        # end of event, change in new param
        duration = last_rec['duration'] + timediff
        # event = 0
        return {'duration': duration, 'event': 0}
    else:
        # put blank parameter
        if last_rec['event'] == 1:
            duration = last_rec['duration'] if last_rec['duration'] is not None else 0 + timediff
        else:
            duration = last_rec['duration'] if last_rec['duration'] is not None else 0
        event = last_rec['event'] if last_rec['event'] else 0
        # print(duration, last_rec['event'])
        return {'duration': duration, 'event': event}


def get_gwc(sm):
    file = open(os.path.join('static', "csv/calibration_data.csv"))
    csv_reader = csv.reader(file)
    rows = []
    for row in csv_reader:
        rows.append({'content': float(row[0]), 'frequency': float(row[1])})

    file.close()
    for it in range(0, len(rows) - 1):
        if rows[it]['frequency'] > sm > rows[it + 1]['frequency']:
            a = rows[it]['frequency'] - rows[it + 1]['frequency']
            b = rows[it + 1]['content'] - rows[it]['content']
            c = a * (rows[it + 1]['frequency']) + b * (rows[it + 1]['content'])
            gwc = (c - (b * sm)) / a
            # print(rows[it][0], gwc)
            return gwc
    return 0


@csrf_exempt
def store_feeds(request):
    if request.method == "POST":
        # store data to db
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        print(json.dumps(body))
        node = Nodes.objects.get(id=body['node_id'])
        if node:
            c_time = datetime.datetime.now(tz=timezone.utc)
            # get leaf wetness duration
            dura = feeds_preprocess(body['node_id'], body['LWS'], c_time)
            print(dura)
            gwc = get_gwc(body['soil_moisture'])
            # get predication for current data
            pred = predict_data(body['temperature'], body['humidity'], body['soil_temperature'], dura['duration'], 0.0)
            pred1= predict_data1(body['temperature'],body['humidity'],body['soil_temperature'],body['LWS'],body['soil_moisture'])
            f_data = Feeds(
                node_id=body['node_id'],
                temperature=body['temperature'],
                humidity=body['humidity'],
                LWS=body['LWS'],
                soil_temperature=body['soil_temperature'],
                soil_moisture=body['soil_moisture'],
                battery_status=body['battery_status'],
                MVP=body['MVP'],
                MVS=body['MVS'],
                SVP=body['SVP'],
                SVS=body['SVS'],
                RO_1=body['RO_1'],
                RO_2=body['RO_2'],
                duration=dura['duration'],
                GWC=gwc,
                event=dura['event'],
                powdery_mildew=pred['powdery_mildew'],
                anthracnose=pred['anthracnose'],
                root_rot=pred['root_rot'],
                irrigation=pred['irrigation'],
                health_status=pred1
            )

            f_data.save()
            node.last_feed_time = c_time
            node.save()
            return HttpResponse(json.dumps(body))

    return HttpResponse()



'''
def store_thingspeak_feeds(node_id, data):
    try:
        # Fetch the node from the database
        node = Nodes.objects.get(id=node_id)
        
        # Debug: Print received data
        #print("Received data:", data)
        
        # Get the current time for the feed
        c_time = datetime.datetime.now(tz=timezone.utc)
        
        # Preprocess the feed data
        print("before preprocess")
        print(data['feeds'][0]['field4'])
        dura = feeds_preprocess(node_id, float(data['feeds'][0]['field4']), c_time)  # Assuming field4 is LWS
        print("after preprocess")
        gwc = get_gwc(float(data['feeds'][0]['field5']))  # Assuming field5 is soil_moisture
        print("after gwc")
        
        # Debug: Print preprocessed data
        print("Preprocessed data:", dura, gwc)

        # Fetch all entries for the current node
        gallery_entries = CropImage.objects.filter(node_id=node_id).order_by('-created_at')

        # Manually filter for the most recent image up to 2 days before the current time
        gallery_entry = next(
            (entry for entry in gallery_entries if c_time.date() >= entry.created_at.date() >= (c_time.date() - datetime.timedelta(days=2))),
            None
        )
        print("is 2 days ka available or not")

        # Fetch the image if available within the date range
        image = None
        if gallery_entry and gallery_entry.image:
            image = gallery_entry.image.read()  # Assuming image is a FileField or ImageField
            print("image milaaa")

        # Predict data with numerical values
        print("Before numerical prediction")
        pred = predict_data(float(data['feeds'][0]['field1']), float(data['feeds'][0]['field2']), float(data['feeds'][0]['field3']), dura['duration'], 0.0)

        # Predict using both numerical and image data if available
        print("Before image-based prediction")
        pred1 = predict_data1(
            data['feeds'][0]['field1'], data['feeds'][0]['field2'], data['feeds'][0]['field3'], data['feeds'][0]['field4'], data['feeds'][0]['field5'], image_file=image
        )

        # Debug: Print predictions
        print("Predictions:", pred, pred1)

        # Create and save feed data
        f_data = Feeds(
            node_id=node_id,
            temperature=float(data['feeds'][0]['field1']),
            humidity=float(data['feeds'][0]['field2']),
            LWS=float(data['feeds'][0]['field4']),
            soil_temperature=float(data['feeds'][0]['field3']),
            soil_moisture=float(data['feeds'][0]['field5']),
            battery_status=float(data['feeds'][0]['field6']),
            MVP=0,
            MVS=0,
            SVP=1,
            SVS=0,
            RO_1=1,
            RO_2=1,
            duration=dura['duration'],
            GWC=gwc,
            event=dura['event'],
            powdery_mildew=pred['powdery_mildew'],
            anthracnose=pred['anthracnose'],
            root_rot=pred['root_rot'],
            irrigation=pred['irrigation'],
            health_status=pred1  # Using result from predict_data1
        )

        f_data.save()

        # Update the node's last feed time
        node.last_feed_time = c_time
        node.save()

        return HttpResponse(json.dumps(data))
    
    except Nodes.DoesNotExist:
        return HttpResponse(status=404, content="Node does not exist.")
    except Exception as e:
        # Debug: Print exception details
        print("Exception occurred:", str(e))
        return HttpResponse(status=500, content=str(e))
'''
@csrf_exempt
def store_thingspeak_feeds(node_id, data):
    try:
        # Fetch the node
        node = Nodes.objects.get(id=node_id)

        # Get the latest entry_id from Feeds for this node
        last_feed = Feeds.objects.filter(node_id=node_id).order_by('-entry_id').first()
        last_entry_id = last_feed.entry_id if last_feed and last_feed.entry_id is not None else 0

        # Filter only new feeds
        new_feeds = [
    f for f in data['feeds']
    if f.get('entry_id') is not None and last_entry_id is not None and int(f['entry_id']) > last_entry_id
]


        if not new_feeds:
            print("No new feeds to process.")
            return HttpResponse("No new data to process.")

        # Get image if available
        gallery_entries = CropImage.objects.filter(node_id=node_id).order_by('-created_at')
        image = None
        now_time = datetime.datetime.now(tz=timezone.utc)
        gallery_entry = next(
            (entry for entry in gallery_entries if now_time.date() >= entry.created_at.date() >= (now_time.date() - datetime.timedelta(days=2))),
            None
        )
        if gallery_entry and gallery_entry.image:
            image = gallery_entry.image.read()

        def parse_float_safe(val):
            try:
                if val is None or str(val).lower() == "inf":
                    return float(0)
                return float(val)
            except (ValueError, TypeError):
                return None

        def parse_int_safe(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return None

        # Process each new feed
        for feed in new_feeds:
            c_time = datetime.datetime.strptime(feed['created_at'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)

            temp = parse_float_safe(feed.get('field1'))
            humid = parse_float_safe(feed.get('field2'))
            soil_temp = parse_float_safe(feed.get('field3'))
            lws = parse_float_safe(feed.get('field4'))
            soil_moist = parse_float_safe(feed.get('field5'))
            battery = parse_float_safe(feed.get('field6'))
            mvp = parse_int_safe(feed.get('field7'))

            # Preprocess duration and GWC
            dura = feeds_preprocess(node_id, lws, c_time)
            gwc = get_gwc(soil_moist)

            # Predict using numerical and image-based models
            pred = predict_data(temp, humid, soil_temp, dura['duration'], 0.0)
            pred1 = predict_data1(temp, humid, soil_temp, lws, soil_moist, image_file=image)

            # Save feed
            f_data = Feeds(
                node_id=node_id,
                entry_id=int(feed['entry_id']),
                created_at=c_time,
                temperature=temp,
                humidity=humid,
                LWS=lws,
                soil_temperature=soil_temp,
                soil_moisture=soil_moist,
                battery_status=battery,
                MVP=mvp,
                MVS=0,
                SVP=1,
                SVS=0,
                RO_1=1,
                RO_2=1,
                duration=dura['duration'],
                GWC=gwc,
                event=dura['event'],
                powdery_mildew=pred['powdery_mildew'],
                anthracnose=pred['anthracnose'],
                root_rot=pred['root_rot'],
                irrigation=pred['irrigation'],
                health_status=pred1
            )
            f_data.save()

            # Update last_feed_time for the node
            node.last_feed_time = c_time
            node.save()

        return HttpResponse("New feeds processed successfully.")

    except Nodes.DoesNotExist:
        return HttpResponse(status=404, content="Node does not exist.")
    except Exception as e:
        print("Exception occurred:", str(e))
        return HttpResponse(status=500, content=str(e))


@login_required
def get_historical_data(request, node_id):
    try:
        # Fetch historical data from the database
        data = Feeds.objects.filter(node_id=node_id).order_by('-created_at')
        
        # Aggregate data for trend analysis (e.g., daily average)
        df = pd.DataFrame(list(data.values()))
        print("data valuesssss")
        #print(data.values())
        df['created_at'] = pd.to_datetime(df['created_at'])
        print(df)
        numeric_df = df.select_dtypes(include=['number'])
        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Aggregate by day, you can also group by weeks or months
        daily_avg = numeric_df.groupby(df['created_at'].dt.date).mean()
        daily_avg.index = daily_avg.index.astype(str)
        print("daily_avggggg")
        print(daily_avg)
        if request.headers.get('x-requested-with') == 'XMLHttpRequest':
            # Return JSON response for AJAX requests
            return JsonResponse(daily_avg.to_dict(orient="index"), safe=False)
        # Return the aggregated data as JSON for use in frontend
        return render(request, 'nodes/historical_data.html', {'node_id': node_id})

    except Feeds.DoesNotExist:
        return JsonResponse({'error': 'No historical data found'}, status=404)



# views.py
# Function to calculate AUDPC
def calculate_audpc(severity_data, time_points):
    audpc = 0
    for i in range(1, len(severity_data)):
        audpc += 0.5 * (time_points[i] - time_points[i - 1]) * (severity_data[i] + severity_data[i - 1])
    return audpc

# Function for linear yield loss calculation
def calculate_linear_yield_loss(severity, max_yield_loss=30):
    yield_loss = (severity / 100) * max_yield_loss
    return yield_loss

# Function for AUDPC-based yield loss calculation
def calculate_audpc_yield_loss(audpc, max_audpc=1000, max_yield_loss=50):
    yield_loss = (audpc / max_audpc) * max_yield_loss
    return yield_loss
def calculate_proxy_severity(lws, temperature, humidity):
    """
    Calculate disease severity using proxy metrics:
    - lws: Leaf Wetness Duration (in hours)
    - temperature: Ambient temperature (in degrees Celsius)
    - humidity: Ambient humidity (in percentage)

    Returns:
        severity: Estimated severity percentage (0 to 100)
    """
    # Define thresholds
    lws_threshold = 10  # Example: LWS > 10 hours is critical
    temp_min, temp_max = 15, 30  # Ideal temperature range for disease
    humidity_threshold = 80  # Humidity > 80% is critical

    # Severity weights
    severity = 0

    # Leaf Wetness Severity
    if lws > lws_threshold:
        severity += min((lws - lws_threshold) * 2, 40)  # Max contribution: 40%

    # Temperature Severity
    if temp_min <= temperature <= temp_max:
        severity += 30  # Ideal temperature range contributes 30%

    # Humidity Severity
    if humidity > humidity_threshold:
        severity += min((humidity - humidity_threshold) * 0.5, 30)  # Max contribution: 30%

    return min(severity, 100)  # Cap severity at 100%

def get_disease_analysis(request, node_id):
    try:
        # Fetch the node and its latest feed data
        node = Nodes.objects.get(id=node_id)
        latest_feed = Feeds.objects.filter(node_id=node_id).order_by('-created_at').first()
        if not latest_feed:
            return JsonResponse({'error': 'No feed data available for this node'}, status=404)

        # Proxy-based severity calculation
        severity = calculate_proxy_severity(
            lws=latest_feed.LWS, 
            temperature=latest_feed.temperature, 
            humidity=latest_feed.humidity
        )
        print(f"Disease Severity (Proxy-Based): {severity}%")

        # Fetch data for disease progression analysis
        feeds = Feeds.objects.filter(node_id=node_id).order_by('created_at')
        time_points = [feed.created_at.timestamp() for feed in feeds]
        severity_data = [
            calculate_proxy_severity(feed.LWS, feed.temperature, feed.humidity) for feed in feeds
        ]

        # Calculate AUDPC
        audpc = calculate_audpc(severity_data, time_points)
        # Normalize AUDPC to a percentage
        normalized_audpc = (audpc / (len(feeds) * 100)) * 100 if len(feeds) > 0 else 0
        print(audpc)
        print(normalized_audpc)

        # Calculate yield loss
        yield_loss_linear = calculate_linear_yield_loss(severity)
        yield_loss_audpc = calculate_audpc_yield_loss(audpc)
        #print("shkhkhkhkhkhkhkhkhkhkhkhkhkhkhkhkdj")
        return render(request, 'nodes/disease_analysis.html', {
        'node_id': node_id,
        'severity': severity,
        'audpc': normalized_audpc,
        'yield_loss_linear': yield_loss_linear,
        'yield_loss_audpc': yield_loss_audpc,
        'time_points': [feed.created_at.strftime('%Y-%m-%d') for feed in feeds] or [],  # Ensure list
        'severity_data': severity_data or []
        })

    except Nodes.DoesNotExist:
        return JsonResponse({'error': 'Node not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@login_required
def get_feeds(request, node_id):
    data = Feeds.objects.filter(node_id=node_id)
    node = Nodes.objects.get(id=node_id)
    return render(request, 'nodes/get_feeds.html', {'data': data, 'node': node, 'node_id': node_id})


@login_required
def get_feeds_table(request, node_id):
    page_num = int(request.GET.get('page', 1))
    if page_num <= 1:
        page_num = 1
    data = Feeds.objects.filter(node_id=node_id).order_by('-id')
    paginator = Paginator(data, 12)
    node = Nodes.objects.get(id=node_id)
    try:
        page_obj = paginator.page(page_num)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)
    return render(request, 'nodes/feed_table.html', {'data': page_obj, 'node': node, 'node_id': node_id})


@login_required
def export_feeds_csv(request, node_id):
    # Define the response object with appropriate headers for a CSV file
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="node_{node_id}_feeds.csv"'

    # Create a CSV writer and write the header row
    writer = csv.writer(response)
    writer.writerow(['Id', 'Node_id', 'Temperature', 'Humidity',
                    'Soil Temperature', 'Soil Moisture', 'LWS', 'Battery', 'Created_at'])

    # Fetch the data and write it to the CSV file
    data = Feeds.objects.filter(node_id=node_id).order_by('-id')

    for feed in data:
        writer.writerow([feed.id, feed.node_id, feed.temperature, feed.humidity, feed.soil_temperature,
                        feed.soil_moisture, feed.LWS, feed.battery_status, feed.created_at.strftime("%b %d, %Y %H:%M:%S")])

    return response


@login_required
def node_list(request):
    # print(predict_data(27.378, 88.05571, 18.84202, 0.5522222222222222, 0.0))
    get_gwc(9687.3379)
    # print(request.user.is_superuser, "--")
    if request.user.is_superuser:
        data = Nodes.objects.all()
    else:
        data = Nodes.objects.filter(user_id=request.user.id)
    date = timezone.now()
    for i in data:
        if (i.last_feed_time is None) or i.last_feed_time is not None and date > i.last_feed_time + datetime.timedelta(
                minutes=30):
            i.status = False
    fetch_data_from_thing_speak(request.user.id)
    return render(request, 'nodes/list.html', {'data': data, 'user_id': request.user.id})


@login_required
def node_particuler_list(request, user_id):
    if not request.user.is_superuser:
        return redirect(to='/get_all_users/')

    data = Nodes.objects.filter(user_id=user_id)
    date = timezone.now()
    for i in data:
        if (i.last_feed_time is None) or i.last_feed_time is not None and date > i.last_feed_time + datetime.timedelta(
                minutes=5):
            i.status = False
    fetch_data_from_thing_speak(request.user.id)
    return render(request, 'nodes/list.html', {'data': data, 'user_id': user_id})


class CrudNodes(View):
    form_class = RegisterForm
    template_name = 'nodes/register.html'

    def dispatch(self, request, *args, **kwargs):
        # will redirect to the home page if a user tries to access the register page while logged in
        if not request.user.is_authenticated:
            return redirect(to='/')

        # else process dispatch as it otherwise normally would
        return super(CrudNodes, self).dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        node_id = int(request.GET.get("id", 0))
        # print(node_id)
        user_id = int(request.GET.get("user_id", 0))
        # print(user_id)
        if node_id != 0:
            data = Nodes.objects.get(id=node_id)
            form = self.form_class(instance=data)
        else:
            form = self.form_class()
        return render(request, self.template_name, {'form': form, 'node_id': node_id, 'user_id': user_id})

    def post(self, request):
        node_id = int(request.POST.get('node_id', 0))
        # print(node_id)
        user_id = int(request.POST.get('user_id', 0))
        # print(user_id)
        if node_id != 0:
            data = Nodes.objects.get(id=node_id)
            form = self.form_class(request.POST, instance=data)
            msg = 'Node Edited successfully'
        else:
            form = self.form_class(request.POST)
            msg = 'Node created successfully'

        if form.is_valid():
            node = form.save(commit=False)
            #node.user_id = request.user.id
            if node.user_id == 0:
                node.user_id = user_id
            if not node.thing_speak_fetch:
                node.channel_id = 0
            node.save()

            messages.success(request, msg)
            if node.user_id == request.user.id:
                return redirect(to='nodes')
            else:
                return redirect(to='/nodes/user_nodes/' + str(node.user_id))

        return render(request, self.template_name, {'form': form})


@login_required
def crop_image_upload(request, node_id):
    form_class = ImageUploadForm
    if request.method == "POST":
        print(request.POST, request.FILES)
        form = form_class(request.POST, request.FILES)
        if form.is_valid():
            crop_image = form.save(commit=False)
            crop_image.node_id = node_id
            crop_image.save()
        messages.success(request, 'Image Upload successfully.')
        return redirect(to='nodes')
    return render(request, 'nodes/image_upload.html', {'form': form_class, 'node_id': node_id})


@login_required
def crop_image_gallery(request, node_id):
    data = CropImage.objects.filter(node_id=node_id)
    print(data)
    node = Nodes.objects.get(id=node_id)
    print(node)
    return render(request, 'nodes/image_gallery.html', {'data': data, 'node': node, 'node_id': node_id})


@login_required
def delete_node(request, node_id):
    # TODO: complete delete feeds code
    node = Nodes.objects.get(id=node_id)
    Feeds.objects.filter(node_id=node_id).delete()
    node.delete()
    messages.success(request, "Node deleted successfully.")
    # return redirect(to='nodes')
    if node.user_id == request.user.id:
        return redirect(to='nodes')
    else:
        return redirect(to='/nodes/user_nodes/' + str(node.user_id))


@login_required
def get_chart_data(request, node_id):
    data = Feeds.objects.filter(node_id=node_id).order_by('-id')
    data1 = data[:200]
    res = serializers.serialize('json', data1)
    return HttpResponse(res, content_type="application/json")

@login_required
def get_last_data(request, node_id):
    try:
        # Fetch node data from the database
        node_data = Nodes.objects.get(id=node_id)
        
        # Check if the node has valid ThingSpeak channel ID and API key
        if not node_data.channel_id or not node_data.node_api_key:
            return JsonResponse({'status': 'error', 'message': 'Node does not have valid ThingSpeak credentials.'}, status=400)
        
        # Construct the URL for ThingSpeak API
        last_feed_url = f"https://api.thingspeak.com/channels/{node_data.channel_id}/feeds.json"
        lf_query = {'api_key': node_data.node_api_key}
        
        # Fetch data from ThingSpeak
        response = requests.get(last_feed_url, params=lf_query)
        
        # Check response status
        if response.status_code == 200:
            data = response.json()
            
            # Call the function to store the feed in the database
            store_thingspeak_feeds(node_id, data)
            return JsonResponse({'status': 'success', 'message': 'Node data refreshed successfully.'})
        else:
            return JsonResponse({'status': 'error', 'message': 'Failed to fetch data from ThingSpeak.'}, status=400)
    
    except Nodes.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Node does not exist.'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)




def fetch_data_from_thing_speak(user_id):
    all_channel = Nodes.objects.filter(user_id=user_id)
    try:
        for channel in all_channel:
            if channel.channel_id is None:
                continue
            last_feed = "https://api.thingspeak.com/channels/" + \
                str(channel.channel_id) + "/feeds.json"
            lf_query = {'api_key': channel.node_api_key, 'minutes': 5}
            response = requests.get(last_feed, lf_query)
            print(data)
            if data['channel']['last_entry_id'] != channel.last_feed_entry:
                # fetch feeds
                # new_feeds = data['channel']['last_entry_id'] - channel.last_feed_entry
                # url = "https://api.thingspeak.com/channels/" + str(channel.id) + "/feeds.json"
                # query = {'api_key': channel.last_entry_id,
                #          'results': new_feeds,
                #          'minutes': 30}
                # response = requests.get(url, query)
                # feeds = response.json()
                for feed in data['feeds']:
                    print(feed)
                    # if type(feed['field5']) != int:
                    #     continue
                    Feeds.objects.create(
                        node_id=channel.id,
                        entry_id=feed['entry_id'],
                        temperature=feed['field1'],
                        humidity=feed['field2'],
                        LWS=feed['field4'],
                        soil_temperature=feed['field3'],
                        soil_moisture=feed['field5'],
                        battery_status=feed['field6'],
                        created_at=feed['created_at']
                    )
                    print("THingspeak Called")

                # update channel
                channel.last_feed_entry = data['channel']['last_entry_id']
                channel.updated_at = timezone.now()
                channel.save()

    except Nodes.DoesNotExist:
        pass


# TODO : image storage in drive
# TODO : backup process
# TODO : 2 way communication



def predict_data1(input1, input2, input3, input4, input5, image_file=None):
    try:
        print("Inside predict_data1 function")
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Construct the path to the model.h5 file
        MODEL_PATH = os.path.join(PROJECT_ROOT, 'static', 'models', 'model.h5')
        print(MODEL_PATH)

        # Load the model
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")

        numeric_inputs = None
        image = None

        # Check if valid numerical inputs are provided
        try:
            numeric_inputs = np.array([[float(input1), float(input2), float(input3), float(input4), float(input5)]])
            print("Numeric inputs shape:", numeric_inputs.shape)
        except ValueError:
            numeric_inputs = None
            print("Invalid or no numerical inputs provided")

        # Check if the image file is provided
        if image_file:
            try:
                image = Image.open(io.BytesIO(image_file))
                image = image.convert('RGB')  # Ensure image is in RGB format
                image = image.resize((224, 224))
                #image = image.resize((2048, 1024))  # Resize image (width, height)
                #image = image.resize((64, 64))
                image = np.array(image) / 255.0  # Normalize the image
                image = np.expand_dims(image, axis=0)  # Add batch dimension
                print("Image input shape:", image.shape)
            except Exception as e:
                print(f"Error processing the image: {e}")
                return "Invalid image file", 400

        # Handle cases based on available inputs
        if numeric_inputs is not None and image is not None:
            # Both numeric and image inputs provided
            result = model.predict([numeric_inputs, image])
        elif numeric_inputs is not None:
            # Only numeric inputs provided, using placeholder for image
            #placeholder_image = np.zeros((1, 1024, 2048, 3))
            placeholder_image = np.zeros((1, 224, 224, 3))
            #placeholder_image = np.zeros((1, 64, 64, 3))
            result = model.predict([numeric_inputs, placeholder_image])
        elif image is not None:
            # Only image input provided, using placeholder for numeric data
            placeholder_numeric = np.zeros((1, 5))
            result = model.predict([placeholder_numeric, image])
        else:
            return "Please provide at least one valid input (numeric or image).", 400

        # Interpret the model's result
        print("Prediction result array:", result)

        if result[0][0] >= 0.5:
            result = 'Leaf Spot Detected'
        elif result[0][1] >= 0.5:
            result = 'Anthracnose Detected'
        else:
            result = 'Healthy'

    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred during prediction.", 500

    return result
    
def predict_data(at, ah, st, lwd, sm):
    arr = [[at, ah, st, lwd, sm]]
    np_arr = np.array(arr)
    df = pd.DataFrame(np_arr,
                      columns=["Ambient_Temperature", "Ambient_Humidity", "Soil_Temperature", "Leaf_Wetness_Duration",
                               "Soil_Moisture"])
    model = keras.models.load_model(os.path.join('static', "models/demo_model.h5"))
    # print(df)
    pred = model.predict(df)
    # print(pred[0], pred[1])
    # return {'powdery_mildew': 0, 'anthracnose': 0, 'root_rot': 0, 'irrigation': 1}
    return {'powdery_mildew': pred[0][0][0], 'anthracnose': pred[0][0][1], 'root_rot': pred[0][0][2],
            'irrigation': pred[1][0][0]}

@login_required
def import_csv(request, node_id):
    form_class = CSVImportForm
    form = form_class()
    if request.method == 'POST':
        form = form_class(request.POST, request.FILES)

        if not form.is_valid():
            messages.error(request, "Invalid form.")
            return redirect(to='nodes')

        node = Nodes.objects.get(id=node_id)
        if not node:
            messages.error(request, "Node not found.")
            return redirect(to='nodes')

        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file)
        df.replace('INF', 0, inplace=True)
        df.drop(['entry_id','latitude','longitude','elevation','status'],axis=1,errors='ignore',inplace=True)

        header_set = set(['created_at','temperature','humidity','soil_temperature','LWS','soil_moisture','battery_status'])

        # check if csv file has exactly same columns headers
        if not set(df.columns) == header_set:
            messages.error(request, "Invalid csv file.")
            return redirect(to='nodes')

        # Insert data into feeds collection
        records = df.to_dict(orient='records')

        # bulk create feed data with adding node id
        for record in records:
            record['node_id'] = node_id
        Feeds.objects.bulk_create([Feeds(**record) for record in records])

        messages.success(request, "data imported successfully.")
        # return redirect(to='nodes')
        if node.user_id == request.user.id:
            return redirect(to='nodes')
        else:
            return redirect(to=f'/nodes/user_nodes/{node.user_id}')

    return render(request, 'nodes/import_csv.html', {'form': form, 'node_id': node_id})

def input_form(request):
    if request.method == 'POST':
        node_id = request.POST.get('nodeId')
        mvp = request.POST.get('mvp')
        mvs = request.POST.get('mvs')
        svp = request.POST.get('svp')
        svs = request.POST.get('svs')
        ro_1 = request.POST.get('ro_1')
        ro_2 = request.POST.get('ro_2')
        try:
            node = Node.objects.get(node_id=node_id)
            # Update existing entry
            node.mvp = mvp
            node.mvs = mvs
            node.svp = svp
            node.svs = svs
            node.ro_1 = ro_1
            node.ro_2 = ro_2
        except Node.DoesNotExist:
            node = Node(
            node_id=node_id,
            mvp=mvp,
            mvs=mvs,
            svp=svp,
            svs=svs,
            ro_1=ro_1,
            ro_2=ro_2
        )
        node.save()
        
        return JsonResponse({
            'nodeId': node.node_id,
            'mvp': node.mvp,
            'mvs': node.mvs,
            'svp': node.svp,
            'svs': node.svs,
            'ro_1': node.ro_1,
            'ro_2': node.ro_2,
        })
    # Fetch data from MongoDB to populate the table
    nodes = Node.objects.all().order_by('-id')
    nodes_json = json.dumps(list(nodes.values()))

    return render(request, 'H_Control/index.html', {'nodes_json': nodes_json})

def get_node_data(request, node_id):
    try:
        node = Node.objects.get(node_id=node_id)
        node_data = {
            'nodeId': node.node_id,
            'mvp': node.mvp,
            'mvs': node.mvs,
            'svp': node.svp,
            'svs': node.svs,
            'ro_1': node.ro_1,
            'ro_2': node.ro_2,
        }
        return JsonResponse(node_data)
    except Node.DoesNotExist:
        return JsonResponse({'error': 'Node not found'}, status=404)
