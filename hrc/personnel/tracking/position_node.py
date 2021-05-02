import os
import yaml
from personnel.tracking.rakf import RAKF3D
from config_file_list import CONFIG_FILES, CONFIG_DIR
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')


class Tag:
    """This class represents a Mobile Position Tag device mounted on the personnel 
    """

    def __init__(self, tag_id, algorithm, filename):
        """Initializes Tag

        Args:
            tag_id (integer): Unique Tag ID
            algorithm (string): The Algorithm used by Tag for processing the positioning data
        """
        try:
            # verify argument
            if (not isinstance(tag_id, int)) and tag_id < 0:
                raise ValueError(
                    "Invalid Tag ID or TagID must me an integer value")
            if algorithm is None:
                raise ValueError("Algorithm must be specified")

            self.tag_id = tag_id
            self.algorithm = algorithm
            self.instance = None
            if algorithm == "RAKF3D":
                self.instance = RAKF3D( filename, tracker_id=tag_id )
            else:
                raise RuntimeError("Algorithm is not supported")
        except RuntimeError as e:
            logging.critical(e)
            exit(-1)
        except ValueError as e:
            logging.critical(e)
            exit(-1)
        except Exception as e:
            logging.critical(e)
            exit(-1)

    def run(self, data):
        """Runs the instance of Algorithm corresponding to the Tag
        """
        try:
            return self.instance.run_on_measurements(data=data)
        except Exception as e:
            logging.critical(e)
            exit(-1)


class Node:
    """This class represents a collection of Tags or a Tag mounted on a Personnel. 
    """

    def __init__(self, name, id, tag_id, algorithm, filename):
        """Initializes personnel class

        Args:
            file_name (with extension .yaml): Describes the name, UUID of personnel, and set of tags mounted on the personnel 
        """
        try:
            if name is not None :
                self.name = name
            else:
                raise ValueError( "Personnel Name missing" )

            if id is not None:
                self.id = id
            else:
                raise ValueError( "Personnel ID missing" )

            if algorithm == "RAKF3D" or algorithm == "RAKF2D":
                self.tag = Tag(tag_id=tag_id, algorithm=algorithm, filename=filename)
            else:
                raise ValueError("Algorithm not supported")
        except ValueError as e:
            logging.critical(e)
            exit(-1)
        except Exception as e:
            logging.critical(e)
            exit(-1)

    def run(self, data):
        try:
            if isinstance(data,dict):
                return self.tag.run(data=data)
            else:
                raise TypeError("Measurement data must be dictionary")
        except TypeError as e:
            logging.critical(e)
            exit(-1)
        except Exception as e:
            logging.critical(e)
            exit(-1)


def create_trackers():
    try:
        walker_file_name = CONFIG_DIR+CONFIG_FILES["personnel"]
        tracker_file_name = CONFIG_DIR+CONFIG_FILES["tracker"]
        personnel_trackers = []
        if os.path.exists( walker_file_name ):
            with open( walker_file_name, 'r' ) as json_file:
                client_json = yaml.load( json_file, Loader=yaml.FullLoader )
                for entry in client_json["personnel_motion"]:
                    if entry["personnel_name"]:
                        name = entry["personnel_name"]
                    else:
                        raise ValueError( "Personnel name missing" )

                    if entry["personnel_id"]:
                        id = entry["personnel_id"]
                    else:
                        raise ValueError( "Personnel id missing" )

                    if entry["walker-attributes"]:
                        walk_dimension = entry["walker-attributes"]["walk-dimension"]
                    else:
                        raise ValueError( "Walk attribute missing in configuration" )

                    if entry["tracker"]:
                        tracker = entry["tracker"]

                        if tracker["tracker_id"]:
                            tagid = tracker["tracker_id"]
                        else:
                            raise ValueError( "tracker-id missing" )

                        if tracker["algorithm"] == 'RAKF':
                            if walk_dimension == 3:
                                # personnel_tracker = RAKF3D( file_name=tracker_file_name, tag_id=tracker["tag_id"] )
                                personnel_tracker = Node( name=name, id=id, tag_id=tagid, algorithm='RAKF3D', filename=tracker_file_name )
                            elif walk_dimension == 2:
                                # personnel_tracker = RAKF3D( file_name=tracker_file_name, tag_id=tracker["tag_id"] )
                                personnel_tracker = Node( name=name, id=id, tag_id=tagid, algorithm='RAKF2D',  filename=tracker_file_name )
                            else:
                                raise ValueError( "Invalid Walk dimension" )
                            personnel_trackers.append( {"id": entry["personnel_id"], "model": personnel_tracker} )
                    else:
                        raise ValueError( "Tracker configuration missing in configuration" )

            return personnel_trackers
        else:
            raise FileNotFoundError( "File not found. Check file name and file path" )
    except FileNotFoundError as e:
        logging.critical( e )
        exit( -1 )
    except Exception as e:
        logging.critical( e )
        exit( -1 )

# ============================ Test ==================================
if __name__ == "__main__":
    person = Node(name="Karsh", id='1234', tracker_id=1, algorithm="RAKF3D")
    print("done")
