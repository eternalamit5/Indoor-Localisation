import asyncio
import json
from matplotlib import animation
from Utilities.DataPlayer import data_player
from Communication.TelegraphMQTTClient import telegraph_mqtt_client
import threading
from Communication.PozyxMQTTClient import pozyx_mqtt_client
from Utilities.PlotAnimation import animate, fig, readData
import matplotlib.pyplot as plt

application_to_run = "DA"


def data_acquisition_mode():
    # create a event loop
    main_event_loop = asyncio.get_event_loop()
    data_player.start_recording()
    # start mqtt client services in separate threads
    threading.Thread( target=telegraph_mqtt_client.run, args=("Config/config.yaml",) ).start()
    threading.Thread( target=pozyx_mqtt_client.run, args=("Config/config.yaml",) ).start()

    # location service as event loop
    # asyncio.ensure_future( person.run() )
    main_event_loop.run_forever()


def data_plot(csv_file_name):
    read_csv_thread = threading.Thread( target=readData, args=(csv_file_name,) ).start()
    ani = animation.FuncAnimation( fig, animate, fargs=(), interval=1 )
    plt.show()
    read_csv_thread.join()


def test():
    data_player.start_recording()
    with open( "Config/data_file.json" ) as f:
        recv_data = json.load(f)
        data_player.record(recv_data)
        data_player.play_from_db(tag_id='2440423245', start_time=1605710762, end_time=1605714368)


if __name__ == '__main__':
    if application_to_run == "DA":
        data_acquisition_mode()
    elif application_to_run == "DP":
        data_plot( "location.csv" )
    else:
        test()