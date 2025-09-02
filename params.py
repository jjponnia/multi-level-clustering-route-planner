import numpy as np

###################################################################################################
# ENVIROMENT
###################################################################################################

########## SIMULATION PARAMS ######################################################################

seed = 2
numAgents = 2 #number of agents spawned on first reset
numTargets = 2 #number of targets spawned on first reset

simParams = {"seed": seed,
             "num_of_agents": numAgents,
             "num_of_targets": numTargets}

########## GRID PARAMS ############################################################################

rows = 10 #200 #number of rows in grid
cols = 10 #200 #number of columns in grid
cellSize = 2000 #25 #size of a single grid element

gridParams = {"rows": rows,
              "cols": cols,
              "cell_size": cellSize}

########## RENDER PARAMS ##########################################################################

plotIDs = False #boolean deciding whether or not to plot the agent ids next to agents in render
manualProgression = False # if True, press space to progress render one iteration at a time

renderParams = {"plot_agent_ids":plotIDs,
                "manual_plot_progression":manualProgression}

###################################################################################################
envParams = {"simParams": simParams,
             "gridParams": gridParams,
             "renderParams": renderParams}
###################################################################################################


###################################################################################################
# AGENTS AND TARGETS
###################################################################################################

########## SPAWN PARAMS ###########################################################################

initialGridPosition = np.array([0, 0]) # agent's starting grid position
lifetime = 100 # number of iterations before agent must despawn and return to initial grid location
randomInit = False # if True, ignore initial grid position and spawn agent in random grid element

spawnParams = {"initial_position": initialGridPosition,
               "lifetime": lifetime,
               "random_init": randomInit}

########## MOVEMENT PARAMS ########################################################################

velocity = 1 # number of elements an agent can jump on the grid in one iteration
movements = {
    0: (0, 1),      # North
    1: (1, 1),      # Northeast
    2: (1, 0),      # East
    3: (1, -1),     # Southeast
    4: (0, -1),     # South
    5: (-1, -1),    # Southwest
    6: (-1, 0),     # West
    7: (-1, 1),     # Northwest
    8: (0, 0)       # Stationary
}

movementParams = {"velocity": velocity,
                  "movements": movements}

########## OBSERVATION PARAMS #####################################################################

aggregationMethod = 'AGE' #priotized method when aggregating new information (AGE vs RESOLUTION)
obsSize = 10 #radius of an agent's local observation
compression = 0.5 #percent confidence lost in information recieved by a single hop between clustered agents

obsParams = {"aggregation_method": aggregationMethod,
             "obs_size": obsSize,
             "compression": compression}

########## CLUSTER PARAMS #########################################################################

# constraints
maxNumAgents = 10 #max number of agents (including cluster head) that can unite under one cluster
l1Distance = 10 #max distance between two agents unclustered agents (chebyshev distance)
radial_buffer = 0.2 # number between 0 and 1.  At (1-radial_buffer) * l1_distance, an agent is an outlier.  At (1+radial_buffer) * l1_distance, the agent must be evicted from the cluster.

# maintenance
escalationLevels = 2 #number of times a cluster head can elevate a transfer request
mergeSize = 1 #max size a cluster can be to merge with its higher cluster

grid_dim = 150 # length and width of grid

# params
clusterParams = {"grid_dim": grid_dim,
                 "max_num_agents": maxNumAgents,
                 "l1_distance": l1Distance,
                 "radial_buffer": radial_buffer,
                 "escalation_levels": escalationLevels,
                 "merge_size": mergeSize}

########## TARGET PARAMS ##########################################################################

targetType = 'UNIFORM' #target distribution (UNIFORM vs CLUSTERED)
numClusters = 3 #number of clusters formed IF target distribution is dense
movement = False #boolean that can introduce movement to targets
strength = 1 #number of agents required to be in the vicinity of the target to clear it off of the grid

# target assignment clustering
alpha = 1 #weight of the distance metric in target assignment clustering
beta = 0.5 #weight of the agent's cluster head in target assignment clustering

targetParams = {"target_type": targetType,
                "num_of_clusters": numClusters,
                "movement": movement,
                "strength": strength,
                "alpha": alpha,
                "beta": beta}

###################################################################################################
agentParams = {"spawnParams": spawnParams,
               "movementParams": movementParams,
               "obsParams": obsParams,
               "clusterParams": clusterParams}
###################################################################################################


###################################################################################################
# ALL PARAMS
###################################################################################################

###################################################################################################
allParams = {"envParams": envParams,
             "agentParams": agentParams,
             "targetParams": targetParams}
###################################################################################################