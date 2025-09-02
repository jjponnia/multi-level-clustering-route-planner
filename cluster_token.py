import numpy as np
# from agent import Agent
from additional_functions import chebyshev_distance
from additional_functions import euclidean

class Agent: pass



class Cluster:
    def __init__(self, level):
        self.level_ = level
        
    def get_level(self):
        return self.level_

class ClusterToken(Cluster):
    def __init__(self, level, clusterParams):
        super().__init__(level)
        
        self.clusterParams = clusterParams
        self.grid_dim = self.clusterParams['grid_dim']

        self.tokenHolder_ = None
        # list of agents
        self.members_ = []

        self.higherCluster_ = None
        self.periodicMaintenanceCounter_ = 0

        self.centroid_ = self.update_centroid() if self.members_ else None

    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)
    
    # def set_token_holder(self,agent): #: Agent):
    #     self.tokenHolder_ = agent

    def get_idle(self):
        if self.level_ == 1:
            return [member for member in self.members_ if not member.clusterHeadToken]
        else:
            idles = []
            for member in self.members_:
                for idle in member.clusterHeadToken.get_idle():
                    idles.append(idle)
            return idles
    
    def get_all_members(self):
        if self.level_ == 1:
            members = [self.tokenHolder_]
            members.extend(self.members_)
            return members
            # return self.members_
        else: 
            allMembers = []
            for member in self.members_:
                # allMembers.append(member)
                # return member.clusterHeadToken.get_all_members()
                for subMember in member.clusterHeadToken.get_all_members():
                    allMembers.append(subMember)
            return allMembers
        
    # def get_all_connections(self):
    #     if self.higherCluster_:
    #         self.higherCluster_.get_all_connections()
    #     else:
    #         allConnections = self.get_all_members()
    #         return allConnections

    def is_outlier(self, agent: Agent):
        self.centroid_ = self.update_centroid()
        dist = euclidean(self.centroid_, agent.position) *  self.grid_dim
        # thresh = self.clusterParams['l1_distance'] * (2**(self.level_ - 1))
        thresh = self.clusterParams['l1_distance'] * (self.level_ + 1)
        # if dist > thresh: pass # print(f'Agent {agent.id} and token holder {self.tokenHolder_.id} have a distance of {dist} and threshold {thresh} with level {self.level_}')
        
        return dist > thresh
    
    # def get_higher_cluster(self):
    #     return self.higherCluster_
        
    # def register_to_head(self, agent: Agent):
    #     # print("Trying to register")
    #     if self.higherCluster_:
    #         self.higherCluster_.join(agent)
    #         # print(f"Registering to head {self.higherCluster_.tokenHolder_.id}")

    # def set_higher_cluster(self, cluster): # : Cluster):
    #     self.higherCluster_ = cluster
    
    def reset_counter(self):
        self.periodicMaintenanceCounter_ = 0

    def cluster_maintenance(self):
        self.reset_counter()
        # outliers = self.members_
        self.select_new_head(False)
        outliers = [member for member in self.members_ if self.is_outlier(member)]

        # filter out any outlier that has the subtree which contains tokenHolder
        outliers = [outlier for outlier in outliers if self.tokenHolder_ not in outlier.get_all_members()]
        if outliers:
            # we should first select a new head and then check for outliers
            # is it possible for the token holder to be in the subtree of an outlier?
            for outlier in outliers:
                # if this returns true, the agent has transfered to another cluster and its local cluster reference has been updated
                if self.check_transfer(outlier, self.tokenHolder_):
                    pass
                else:
                    radial_buffer = self.clusterParams['radial_buffer']
                    distance = euclidean(outlier.position, self.centroid_) * self.grid_dim
                    if distance > (1 + radial_buffer) * self.clusterParams['l1_distance'] * (self.level_ + 1):
                        # need to remove the outlier from the list of cluster members
                        members = self.members_
                        new_member_list = [member for member in members if member.id != outlier.id]
                        self.members_ = new_member_list
                        # need to remove the token reference from the outlier and this depends on whether or not the token is level 1
                        if self.level_ == 1:
                            outlier.lvl1Cluster = None
                        else:
                            outlier.clusterHeadToken.higherCluster_ = None
        # apparently need to add stuff?

        if len(self.members_) >= self.clusterParams['max_num_agents']:
            self.split_cluster()
        
        if self.higherCluster_:
            # if there are too few members in the cluster then we need to merge
            if len(self.members_) <= self.clusterParams['merge_size']:
                # if a cluster is too small, it gets merged with its higher cluster
                # we pass the current token holder into the request_merge function of the higher cluster token
                self.higherCluster_.request_merge(self.tokenHolder_)
        # if there is no higher cluster and there is only one member destroy the token
        elif len(self.members_) <= 1 and self.level_ > 1:
            # deregister each member from the token
            for member in self.members_:
                member.clusterHeadToken.higherCluster_ = None

            self.tokenHolder_.clusterHeadToken = None

            # token = self.tokenHolder_.release_token()
            # token.destroy_token()

    def agg_information(self):
        # Upstream aggregation (members providing information to token holders)
        for member in self.members_:
            if member.id != self.tokenHolder_.id:
                self.tokenHolder_.aggView_.aggregate(member.aggView_,transfer=True)
        
        # Downstream aggregation (token holders providing information to members)
        for member in self.members_:
            if member.id != self.tokenHolder_.id:
                member.aggView_.aggregate(self.tokenHolder_.aggView_,transfer=True)
    
    # def add_to_cluster(self, agent: Agent):
    #     if agent.id not in [member.id for member in self.members_]:
    #         self.members_.append(agent)

    #     if self.level_ == 1:
    #         agent.set_lvl1_cluster(self)
    #     else:
    #         agent.set_higher_cluster(self)
    #     # print(f"Agent {agent.id} joined cluster headed by {self.get_cluster_head_id()}")

    def select_new_head(self, force):
        self.centroid_ = self.update_centroid() # Calculating centroid
        candidates = self.get_idle() # .get_idle()

        if not force:
            candidates.append(self.tokenHolder_)

        if not candidates:
            return False

        centroidDistance = [(candidate, self.centroid_distance(candidate)) for candidate in candidates]
        sortedCandidateList = sorted(centroidDistance, key=lambda x: x[1])
        bestAgent = sortedCandidateList[0][0]

        # this if statement is weird - checks if there is an existing token holder.  But there always should be one if this function is called
        # if self.tokenHolder_ is not None:
        if self.tokenHolder_.id == bestAgent.id:
            return False
            
        if not force and self.centroid_distance(self.tokenHolder_) < self.centroid_distance(bestAgent) + 1e-5:
            return False

        current_token_holder = self.tokenHolder_

        # remove the current token holder from the higher cluster if it exists and add bestAgent instead
        if self.higherCluster_:
            members = self.higherCluster_.members_
            # remove current token holder from the higher cluster
            new_member_list = [member for member in members if member.id != current_token_holder.id]
            # add bestAgent to the higher cluster
            new_member_list.append(bestAgent)
            # update the higher cluster's members list
            self.higherCluster_.members_ = new_member_list

        # take the token away from the current token holder
        current_token_holder.clusterHeadToken = None

        # assign the token to the best agent
        self.tokenHolder_ = bestAgent
        bestAgent.clusterHeadToken = self

        return True


    def update_centroid(self):
        # self.upgrade_members()
        sumLoc = self.tokenHolder_.position
        for member in self.members_:
            sumLoc = [x + y for x, y in zip(sumLoc, member.position)]


        return [x / (len(self.members_) +  1) for x in sumLoc]


    def centroid_distance(self, agent: Agent):
        return euclidean(self.centroid_, agent.position)
    
    # def upgrade_members(self):
    #     # assumes all elements in self.members_ are NOT None
    #     return self.members_
    
    # def deregister_from_head(self, id):
    #     if self.higherCluster_:
    #         self.higherCluster_.despawn(id)

    # def destroy_token(self):
    #     if self.higherCluster_:
    #         if self.tokenHolder_:
    #             self.higherCluster_.despawn(self.tokenHolder_.id)
    #         else:
    #             pass # print("Token doesn't have token holder? Interesting. . .")
        
    #     for member in self.members_:
    #         if self.level_ == 1:
    #             member.evict_from_lvl1()
    #         else:
    #             member.evict_from_cluster()

    # def get_cluster_head_id(self):
    #     if self.tokenHolder_:
    #         return self.tokenHolder_.id
    #     else:
    #         return None
    
    def is_full(self):
        return len(self.members_) >= self.clusterParams["max_num_agents"]

    def request_transfer(self, agent: Agent, previousHead: Agent):
        level = previousHead.clusterHeadToken.level_
        members = []
        # the "self" here is the higher cluster of the previous head
        for member in self.get_lvlX_members(level):
            if not member.clusterHeadToken.is_full():
                members.append(member)

        members.sort(key=lambda member: euclidean(agent.position, member.position))
        # since the agent being transfered is already in a cluster, members should never be empty
        if members:
            # the best head for the outlier is the same as the previous head
            if members[0].id == previousHead.id:
                # escalate the request to a higher cluster
                if self.higherCluster_:
                    # we are out of escalation levels
                    if self.higherCluster_.level_ > level + self.clusterParams["escalation_levels"]:
                        return False
                    else:
                        return self.higherCluster_.request_transfer(agent, previousHead)
                else:
                    # if no higher cluster exists then the transfer fails
                    return False
            # best head for the outlier is different from the previous head
            else:
                # remove agent from the current cluster token
                current_cluster_token = previousHead.clusterHeadToken
                members = current_cluster_token.members_
                new_member_list = [member for member in members if member.id != agent.id]
                current_cluster_token.members_ = new_member_list

                # add the agent to the new cluster token
                new_cluster_token = members[0].clusterHeadToken
                new_cluster_token.members_.append(agent)

                # update the cluster head token of the agent (it has a new higher cluster)
                if new_cluster_token.level_ == 1:
                    agent.lvl1Cluster = new_cluster_token
                else:
                    agent.clusterHeadToken.higherCluster_ = new_cluster_token

                return True
        else:
            # no candidates available for transfer
            return False

    def check_transfer(self, agent: Agent, previousHead: Agent):
        # print(f'Checking transfer for agent {agent.id} with cluster head {previousHead.id}')
        if not self.higherCluster_:
            return False
        else:
            return self.higherCluster_.request_transfer(agent, previousHead)

    def split_cluster(self):
        # self.centroid_ = self.update_centroid()
        idles = self.get_idle()

        th = self.tokenHolder_
        idles.append(th)

        if len(idles) == 1:
            return False
        else:
            idles.sort(key=lambda agent: self.centroid_distance(agent))

        newHead = idles[1] if idles[0].id == th.id else idles[0]
        newtoken = ClusterToken(level = self.level_, clusterParams = self.clusterParams)
        newtoken.tokenHolder_ = newHead

        for member in self.members_:
            # we go through all members to ensure that every clusterhead belongs to a level-1 cluster in its subtree
            allMembers = member.get_all_members()

            # if the token holder is a child of member, then don't put member in the new cluster
            if th in allMembers:
                continue

            # if newHead is not a child of member, member can still be added to the new cluster provided member is closer to the token holder than to newHead
            if newHead not in allMembers:
                # if member is closer to the token holder than to newHead, then don't put member in the new cluster
                if euclidean(member.position, th.position_) <= euclidean(member.position, newHead.position):
                    continue

            # if newHead is a child of member, then member should definitely be added to the new cluster
            # what if newHead is member (can happen if current cluster is a level 1)?  don't add it to newMembers
            if newHead.id != member.id:
                newtoken.members_.append(member)
                if self.level_ == 1:
                    member.lvl1Cluster = newtoken
                else:
                    member.clusterHeadToken.higherCluster_ = newtoken

        if not newtoken.members_:
            return False

        # the members that have been added to the new cluster should be removed from the old cluster
        updatedMembers = [member for member in self.members_ if member not in newHead.members_ and member.id != newHead.id]
        self.members_ = updatedMembers

        if self.level_ == 1:
            # if the new head is a level 1 cluster, then it should be the token holder of the new token
            newHead.lvl1Cluster = newtoken
        else:
            # if the new head is a higher level cluster, then it should be the token holder of the new token
            newHead.clusterHeadToken = newtoken

        # add newHead to the higherCluster if higherCluster isn't full
        if not self.higherCluster_.is_full():
            self.higherCluster_.members_.append(newHead)
            newtoken.higherCluster_ = self.higherCluster_

            
    def request_merge(self, agent: Agent):
        # get the number of members in the cluster requesting a merge
        num_merge_members = len(agent.clusterHeadToken.members_) + 1 # + 1 for the agent itself
        merge_candidates = []

        # I want to look through the cluster members and see if there is a cluster with enough free slots to merge with the requestor
        for member in self.members_:
            # print(f'Checking member {member.id} with cluster head {member.clusterHeadToken.get_cluster_head_id()}')
            if member.id != agent.id and member.clusterHeadToken.get_num_free_slots() >= num_merge_members:
                merge_candidates.append(member)

        # get the member closest to the agent requesting a merge
        merge_candidates.sort(key=lambda member: euclidean(agent.position, member.position))

        if merge_candidates:
            newHead = merge_candidates[0]
        else:
            return False

        # move all the agents in the cluster requesting a merge to the new head's cluster
        oldtoken = agent.clusterHeadToken
        newToken = newHead.clusterHeadToken

        for member in oldtoken.members_:
            # newHead.clusterHeadToken.add_to_cluster(member)
            newToken.members_.append(member)
            if newToken.level_ == 1:
                member.lvl1Cluster = newHead.clusterHeadToken
            else:
                member.clusterHeadToken.higherCluster_ = newHead.clusterHeadToken

        # delete agent from the old token's higher level cluster
        oldHigherToken = agent.clusterHeadToken.higherCluster_
        oldHigherToken.members_ = [member for member in oldHigherToken.members_ if member.id != agent.id]

        # if the old token is a level 1 token, then add agent to the new level 1 token
        if newToken.level_ == 1:
            newToken.members_.append(agent)
            agent.lvl1Cluster = newToken
        # if the old token is a higher level token then delete token from the agent
        else:
            agent.clusterHeadToken = None

        return True

    def get_num_free_slots(self):
        max_num_agents = self.clusterParams['max_num_agents']
        if max_num_agents <= len(self.members_):
            return 0
        else:
            return max_num_agents - len(self.members_)

    def get_lvlX_members(self, level):
        if level == self.get_level() - 1:
            return self.members_
        
        if level < self.level_ - 1:
            members = self.members_
            lvlXMembers = []
            for member in members:
                ms = member.clusterHeadToken.get_lvlX_members(level)
                lvlXMembers.extend(ms) # extend adds multiple elements to a list vs append adds one element
            return lvlXMembers
        return []
    
    # def join(self,agent: Agent):
    #     self.add_to_cluster(agent)
    #     return True
    
    # def despawn(self, agent_id: int):
    #     self.remove_from_cluster(agent_id, True)

    # def remove_from_cluster(self, id: int, evict: bool):
    # def remove_from_cluster(self, agent_id):
    #     # print(f"Attempting to remove agent {id} from cluster headed by {self.tokenHolder_.id}")

    #     # find the member with the given id
    #     m = next((member for member in self.members_ if member.id == agent_id), None)
    #     # remove the token reference from the agent
    #     if self.level_ == 1:
    #         m.lvl1Cluster = None
    #     else:
    #         m.clusterHeadToken = None
    #     # remove the agent from the members list
    #     self.members_ = [member for member in self.members_ if member.id != agent_id]

    #     # if not m:
    #         # print("Member not found in self.members_")
    #     #     return False
    #     # else:

    #     # if evict:
    #     #     if self.level_ == 1:
    #     #         m.evict_from_lvl1()
    #     #     else:
    #     #         m.evict_from_cluster()

