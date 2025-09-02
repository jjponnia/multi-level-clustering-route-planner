import csv
import os
import math
import numpy as np

# additional_functions
def euclidean(p1, p2):
    x1,y1 = p1[0],p1[1]
    x2,y2 = p2[0],p2[1]
    
    # return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return abs(x1-x2)+abs(y1-y2)

def chebyshev_distance(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]

    return max(abs(x2 - x1), abs(y2 - y1))

def matrix_blocks(array_list):

    array = array_list[0]

    for i in range(1,len(array_list)):
        nx, ny = array.shape # current size of array
        nix, niy = array_list[i].shape # next block of ones size

        array = np.block([
            [array, np.zeros(shape=(nx,niy))],
            [np.zeros(shape=(nix,ny)), array_list[i]]
        ])
    
    return array

def write_dict_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path)

    try:
        with open(file_path, 'a', newline='') as csvfile:
            # Extract the keys (column headers) from the dictionary
            fieldnames = list(data.keys())

            # Create a CSV writer object with the fieldnames if the file is new
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write the header row only if the file is new
            if not file_exists:
                csv_writer.writeheader()

            # Write data to the CSV file
            csv_writer.writerow(data)

            # print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

class AgentPosition:
    def __init__(self, position, age, remaining_life):
        self.position_ = position
        self.age = age
        self.remaining_life = remaining_life

class AgentCell:
    def __init__(self, occ):
        self.occupied = occ # bool
        
        self.agents = []
        self.agent_ids = []
        
    def add_agent(self, agent):
        self.agents.append(agent)
        self.agent_ids.append(agent.id)

class TargetCell:
    def __init__(self, present):
        self.present = present
        self.targets = []
        self.target_ids = []

    def add_target(self, target):
        self.targets.append(target)
        self.target_ids.append(target.id)