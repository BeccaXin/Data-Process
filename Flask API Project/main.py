# python3 main.py georef-australia-state-suburb.csv au.csv

from flask import Flask, request,jsonify, make_response
from flask_restful import Api, Resource, reqparse, fields, marshal_with
from flask_restx import Api, Resource, fields
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import geopandas as gpd
import seaborn as sns
import datetime
import requests
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import io
import threading
import sys
import sqlite3

matplotlib.use('Agg')

app = Flask(__name__)
api = Api(app,default="Events",title="Events",description="REST API MyCalendar")

events=[]

with open('database.json', 'w') as f:
    json.dump(events, f)
    

event_post_model = api.model('Event', {
    'name': fields.String(required=True),
    'date': fields.Date(required=True,format='%d-%m-%Y'),
    'from': fields.String(required=True),
    'to': fields.String(required=True),
    'location': fields.Nested(api.model('Location', {
        'street': fields.String(required=True),
        'suburb': fields.String(required=True),
        'state': fields.String(required=True),
        'post-code': fields.String(required=True)
    })),
    'description': fields.String()
})


event_patch_model = api.model('Event Patch Model', {
    'name': fields.String,
    'date': fields.Date(format='%d-%m-%Y'),
    'from': fields.String,
    'to': fields.String,
    'location': fields.Nested(api.model('Location', {
        'street': fields.String,
        'suburb': fields.String,
        'state': fields.String,
        'post-code': fields.String
    })),
    'description': fields.String()
})


@api.route('/events')
class Events(Resource):
    @api.doc(description='Create a new event,please date input follow dd-mm-yyyy and from input follow hh:mm:ss')
    @api.expect(event_post_model, validate=True)
    def post(self):
        new_event_post = request.json
        
        for event in events:
            if event['date'] == new_event_post['date']:
                if new_event_post['from'] < event['to'] and new_event_post['to'] > event['from']:
                    return {'message': 'Event overlaps with existing event'}, 400


        with open('z5386659_database.json', 'w') as f:
            json.dump(events, f)
            
        new_event_id = len(events) + 1
        new_event_post['id'] = new_event_id
        new_event_post['last-update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        events.append(new_event_post)

        response = {
            'id': new_event_id,
            'last-update': new_event_post['last-update'],
            '_links': {
                'self': {
                    'href': '/events/' + str(new_event_id)
                }
            }
        }
        
 
        with open('z5386659_database.json', 'r') as f:
            sql_event_data = json.load(f)
 
                
        sql_events_file = sqlite3.connect('Z5386659.db')
        sql_events_table = sql_events_file.cursor()

        sql_events_table.execute('''CREATE TABLE IF NOT EXISTS Events
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      date TEXT,
                      from_ TEXT,
                      to_ TEXT,
                      street TEXT,
                      suburb TEXT,
                      state TEXT,
                      postcode TEXT,
                      description TEXT)''')

        for event in sql_event_data:
            sql_events_table.execute('''INSERT INTO events (name, date, from_, to_, street, suburb, state, postcode, description)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (event['name'], event['date'], event['from'], event['to'],
                       event['location']['street'], event['location']['suburb'], event['location']['state'], event['location']['post-code'],
                       event['description']))

        sql_events_file.commit()
        sql_events_file.close()
        
        return response, 201
        
#/events?order=+id&page=1&size=10&filter=id,name
#python3 Z5386659.py georef-australia-state-suburb.csv au.csv


    def get(self):
 
        with open('z5386659_database.json', 'r') as f:
            events = json.load(f)

        order = request.args.get('order', '+id')
        page = int(request.args.get('page', 1))
        size = int(request.args.get('size', 10))
        filter = request.args.get('filter', 'id,name')

        order_input_list = []
        order_1=order.split(',')
        for order_input in order_1:
            if order_input.startswith('+'):
                order_input_list.append((order_input[1:], True))
            elif order_input.startswith('-'):
                order_input_list.append((order_input[1:], False))
            else:
                order_input_list.append((order_input, True))

        def sort_events(events):
            for order_input, ascend in order_input_list:
                if order_input == 'id':
                    events = sorted(events, key=lambda x: int(x.get('id')), reverse=not ascend)
                elif order_input == 'name':
                    events = sorted(events, key=lambda x: x.get('name').lower(), reverse=not ascend)
                elif order_input == 'date' or order_input == 'datetime':
                    events = sorted(events, key=lambda x: datetime.datetime.strptime(x.get('date') + " " + x.get('from'), '%d-%m-%Y %H:%M:%S'), reverse=not ascend)
            return events

        events = sort_events(events)
        
        for order_input, ascend in order_input_list:
            if ascend:
                events = list(events)


#
        filter_input = filter.split(',')
        events = [{input: e[input] for input in filter_input} for e in events]

        start_page = (page - 1) * size
        end_page = start_page + size
        events_page = events[start_page:end_page]

 
        response = {
             'page': page,
             'page-size': size,
             'events': events_page,
             '_links': {
                 'self': {'href': '/events?order={}&page={}&size={}&filter={}'.format(order, page, size, filter)},
             }
            }

        if end_page < len(events):
            response['_links']['next'] = {'href': '/events?order={}&page={}&size={}&filter={}'.format(order, page+1, size, filter)}

        return response, 200


@api.route('/events/<int:event_id>')
class Event_id(Resource):

    @api.doc(description='Get an event by id')
    def get(self, event_id):
        for event in events:
            if event['id'] == event_id:
                response = {
                    'id': event['id'],
                    'last-update': event['last-update'],
                    'name': event['name'],
                    'date': event['date'],
                    'from': event['from'],
                    'to': event['to'],
                    'location': event['location'],
                    'description': event['description'],
                    '_metadata':{},
                    '_links': {}
                }
                
                sorted_events = sorted(events, key=lambda a: (a['date'][6:], a['date'][3:5], a['date'][:2], a['from']))

                
                index = sorted_events.index(event)
                
                if index:
                    response['_links']['self'] = {
                        'href': f"/events/{sorted_events[index]['id']}"
                    }
                if index > 0:
                    response['_links']['previous'] = {
                        'href': f"/events/{sorted_events[index-1]['id']}"
                    }
                if index < len(sorted_events) - 1:
                    response['_links']['next'] = {
                        'href': f"/events/{sorted_events[index+1]['id']}"
                    }
                
                if "suburb" in event["location"]:
                    suburb = event["location"]["suburb"]
                    if suburb is not None:
                        if suburb in lat_lng_data_city:
                            dateff=event['date']
                            lat, lng = lat_lng_data_city[suburb]
                            hour_weather=event['from']
                            weather_data = get_weather_data(lat, lng, dateff, hour_weather)
                            response["_metadata"] = weather_data
                        elif suburb in lat_lng_data_suburb:
                            lat, lng = lat_lng_data_suburb[suburb]
                            dateff=event['date']
                            hour_weather=event['from']
                            weather_data = get_weather_data(lat, lng, dateff, hour_weather)
                            response["_metadata"] = weather_data

                return response,200
        return {'message': 'Event not found'}, 404
        

    @api.doc(description='Delete an event by id')
    def delete(self, event_id):
        for event in events:
            if event['id'] == event_id:
                events.remove(event)

                return {'message': 'The event with id {} was removed from the database!'.format(event_id)}, 200
        return {'message': 'Event not found'}, 404
        
    @api.doc(description='Update events by id, please date input follow dd-mm-yyyy and from input follow hh:mm:ss')
    @api.expect(event_patch_model)
    def patch(self, event_id):
        for event in events:
            if event['id'] == event_id:
                event.update(request.json)
                event['last-update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                response = {
                    'id': event_id,
                    'last-update': event['last-update'],
                    '_links': {
                        'self': {'href': '/events/{}'.format(event_id)}
                    }
                }
                return response, 200
        return {'message': 'Event not found'}, 404


@api.route('/events/statistics')
class Statistics(Resource):
    def get(self):

        with open('z5386659_database.json', 'r') as f:
            events = json.load(f)

        format = request.args.get('format', 'json')

        today = datetime.date.today()
        current_week_start = today - datetime.timedelta(days=today.weekday())
        current_month_start = datetime.date(today.year, today.month, 1)
        events_per_day = {}
        
        total_count = 0
        current_week_count = 0
        current_month_count = 0
        for event in events:
            date_check = event['date']
            event_date = datetime.datetime.strptime(date_check, '%d-%m-%Y').date()
            
            if event_date in events_per_day:
                events_per_day[event_date] += 1
            else:
                events_per_day[event_date] = 1
            total_count += 1
            
            if current_week_start <= event_date <= today:
                current_week_count += 1
            if current_month_start <= event_date <= today:
                current_month_count += 1

        if format == 'json':
            response = {
                'total': total_count,
                'total-current-week': current_week_count,
                'total-current-month': current_month_count,
                'per-days': {date.isoformat(): count for date, count in events_per_day.items()}
            }
            return response, 200


        elif format == 'image':
            fig, ax = plt.subplots()
            ax.plot(events_per_day.keys(), events_per_day.values())
            ax.set_xlabel('Date: Year-month-day')
            ax.set_ylabel('Number of events')
            ax.set_title('Number of events per month')

            img_event = io.BytesIO()
            fig.savefig(img_event, format='png')
            img_event.seek(0)

            response = make_response(img_event.getvalue())
            response.headers.set('Content-Type', 'image/png')
            return response
        else:
            return {'error': 'Invalid format parameter'}, 400

api.add_resource(Event_id, '/events/<int:event_id>')
api.add_resource(Statistics, '/events/statistics')

@api.route('/weather')
class Weather(Resource):
    def get(self):
        date_input = request.args.get('date')
        date_weather_Aus=date_input.split('-')
        date_weather_Aus.reverse()
        date_weather_Aus = ''.join(date_weather_Aus)
        
        #prepare to choose top15 cities
        cities = pd.read_csv(input_2).sort_values('population', ascending=False).head(15)

        weather_data = []
        for _, city in cities.iterrows():
            lat, lng = city[['lat', 'lng']]
            url_weather = f"https://www.7timer.info/bin/civil.php?lat={lat}&lng={lng}&ac=1&unit=metric&output=json&product=two&date={date_weather_Aus}"
            response = requests.get(url_weather)
            if response.status_code == 200:
                data = response.json()
                weather = data['dataseries'][0]['weather']
                temperature = data['dataseries'][0]['temp2m']
                weather_data.append({
                    'city': city['city'],
                    'lat': lat,
                    'lng': lng,
                    'weather': weather,
                    'temperature': temperature
                })
            else:
                print(f"Error matching weather data for {city['city']}")

        weather_df = pd.DataFrame(weather_data)

#This map file I search and get from this website:https://stackoverflow.com/questions/75668431/plotting-points-on-australia-map-in-python
        
        australia_map = gpd.read_file('https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/STE_2021_AUST_SHP_GDA2020.zip')

 
        cities_geo = gpd.GeoDataFrame(
            weather_df, geometry=gpd.points_from_xy(weather_df.lng, weather_df.lat), crs="EPSG:4326"
        ).to_crs(australia_map.crs)
        
        cities_merged = gpd.sjoin(cities_geo, australia_map, op='within').drop(columns='geometry')

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(15, 15))

        sns.scatterplot(data=cities_merged, x='lng', y='lat', hue='city', s=100, legend='full', zorder=2)

        for _, row in cities_merged.iterrows():
            ax.annotate(
                f" {row['temperature']}°C",
                xy=(row['lng'], row['lat']),
                ha='left',
                va='center',
                fontsize=13
            )

        australia_map.plot(ax=ax, color='lightgrey', edgecolor='white', zorder=1)

        plt.title('Weather of top 15 population cities in Australia today')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        img_AUS_weather = io.BytesIO()
        plt.savefig(img_AUS_weather, format='png')
        img_AUS_weather.seek(0)
        
        response = make_response(img_AUS_weather.getvalue())
        response.headers['Content-Type'] = 'image/png'
        
        return response



if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Wrong inputs format, try again")
        sys.exit(-1)
        
    input_1=sys.argv[1]
    input_2=sys.argv[2]
    
    
    
    with open('z5386659_database.json') as f:
        events = json.load(f)

    url_holiday = 'https://date.nager.at/api/v2/publicholidays/2023/AU'
    response = requests.get(url_holiday)
    holidays = response.json()
    
    state_postcode = {
    'NSW': '2000-2999',
    'NT': '0800-0999',
    'QLD': '4000-9999',
    'ACT': '2600-2920',
    'SA': '5000-5999',
    'TAS': '7000-7999',
    'VIC': '3000-8999',
    'WA': '6000-6999'}
    
    for holiday in holidays:
        holiday_date = holiday['date']
        formatted_date = holiday['date'].split('-')
        formatted_date.reverse()
        formatted_date = '-'.join(formatted_date)
        holiday['date'] = formatted_date
        counties = holiday.get('counties')
        state = ''
        postcode = ''
        if counties:
            county = counties[0]
            state = county[3:]
            postcode= state_postcode.get(state)
            
        event_holiday = {
        'name': holiday['name'],
        'date': holiday['date'],
        'from': '02:00:00',
        'to': '07:00:00',
        'location': {
            'street': '',
            'suburb': '',
            'state': state,
            'post-code': postcode
        },
        'description': 'Holiday'
    }
        event_holiday['id'] = len(events) + 1
        event_holiday['last-update'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        events.append(event_holiday)
    
    
    def get_lat_lng_suburb_data(file):
        lat_lng_data_suburb = {}
        with open(file) as suburb_csv_file:
            reader = csv.DictReader(suburb_csv_file)
            for row in reader:
                suburb=row["Official Name Suburb"]
                lat_lng = row["Geo Point"].split(",")
                if len(lat_lng) == 2:
                    lat_lng_data_suburb[suburb] = (lat_lng[0], lat_lng[1])
        return lat_lng_data_suburb

    
    
    def get_lat_lng_city_data(file):
        lat_lng_data_city = {}
        with open(file) as city_csv_file:
            reader = csv.DictReader(city_csv_file)
            for row in reader:
                lat_lng_data_city[row["city"]] = (row["lat"], row["lng"])
        return lat_lng_data_city
    

    def get_weather_data(lat, lng, date, hour):
        date_weather=date.split('-')
        date_weather.reverse()
        date_weather = ''.join(date_weather)
        url_weather = f"https://www.7timer.info/bin/civil.php?lat={lat}&lng={lng}&ac=1&unit=metric&output=json&product=two&date={date_weather}&time={hour}"
        response = requests.get(url_weather)
        if response.status_code == 200:
            weather_data = response.json()
            
            for event in events:
                if event['description'] == 'holiday':
                    metadata = {
                        "wind-speed": f"{weather_data['dataseries'][0]['wind10m']} KM",
                        "weather": weather_data['dataseries'][0]['weather'],
                        "humidity": f"{weather_data['dataseries'][0]['rh2m']}",
                        "temperature": f"{weather_data['dataseries'][0]['temp2m']} C",
                        "holiday": event['name'],
                        "weekend": datetime.datetime.strptime(date, "%d-%m-%Y").weekday() >= 5
                    }
                    return metadata

                metadata = {
                    "wind-speed": f"{weather_data['dataseries'][0]['wind10m']['speed']} KM",
                    "weather": weather_data['dataseries'][0]['weather'],
                    "humidity": f"{weather_data['dataseries'][0]['rh2m']}",
                    "temperature": f"{weather_data['dataseries'][0]['temp2m']} C",
                    "holiday": None,
                    "weekend": datetime.datetime.strptime(date, "%d-%m-%Y").weekday() >= 5
                }
                return metadata
        else:
            return None


    df = pd.read_csv(input_1, sep=";")
    suburb_df = df[["Geo Point", "Official Name Suburb"]]
    suburb_df.to_csv("georef-australia-state-suburb_new.csv", index=False)
    
    lat_lng_data_suburb = get_lat_lng_suburb_data("georef-australia-state-suburb_new.csv")
    lat_lng_data_city = get_lat_lng_city_data(input_2)
   
    app.run(debug=True)
    
