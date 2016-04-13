#!/usr/bin/env python3

########################################################################################
#
# Name: PCFG_Cracker "Next" Function
# Description: Section of the code that is responsible of outputting all of the
#              pre-terminal values of a PCFG in probability order.
#              Because of that, this section also handles all of the memory management
#              of a running password generation session
#
#########################################################################################


import sys   #--Used for printing to stderr
import string
import struct
import os
import types
import time
import queue
import copy
import heapq

from pcfg_manager.ret_types import RetType
from pcfg_manager.core_grammar import PcfgClass

###################################################################################################
# Used to hold the parse_tree of a path through the PCFG that is then stored in the priority queue
###################################################################################################
class QueueItem:
    
    ############################################################################
    # Basic initialization function
    ############################################################################
    def __init__(self, is_terminal = False, probability = 0.0, parse_tree = []):
        self.is_terminal = is_terminal      ##-Used to say if the parse_tree has any expansion left or if all the nodes represent terminals
        self.probability = probability      ##-The probability of this queue items
        self.parse_tree = parse_tree        ##-The actual parse through the PCFG that this item represents
        
        
    ##############################################################################
    # Need to have a custom compare functions for use in the priority queue
    # Really annoying that I have to do this the reverse of what I'd normally expect
    # since the priority queue will output stuff of lower values first.
    # Aka if there are two items with probabilities of 0.7 and 0.4, the PQueue will
    # by default output 0.4 which is ... not what I'd like it to do
    ##############################################################################
    def __lt__(self, other):
        return self.probability > other.probability
    
    def __le__(self, other):
        return self.probability >= other.probability
        
    def __eq__(self, other):
        return self.probability == other.probability
        
    def __ne__(self, other):
        return self.probability != other.probability
        
    def __gt__(self, other):
        return self.probability < other.probability
        
    def __ge__(self, other):
        return self.probability <= other.probability
    
    
    ###############################################################################
    # Overloading print operation to make debugging easier
    ################################################################################
    def __str__(self):
        ret_string = "isTerminal = " + str(self.is_terminal) + "\n"
        ret_string += "Probability = " + str(self.probability) + "\n"
        ret_string += "ParseTree = " + str(self.parse_tree) + "\n"
        return ret_string
       
       
    #################################################################################
    # A more detailed print that is easier to read. Requires passing in the pcfg
    #################################################################################
    def detailed_print(self,pcfg):
        ret_string = "isTerminal = " + str(self.is_terminal) + "\n"
        ret_string += "Probability = " + str(self.probability) + "\n"
        ret_string += "ParseTree = "
        ret_string += pcfg.print_parse_tree(self.parse_tree)
        return ret_string
        
    
            
#######################################################################################################
# I may make changes to the underlying priority queue code in the future to better support
# removing low probability items from it when it grows too large. Therefore I felt it would be best
# to treat it as a class. Right now though it uses the standared python queue HeapQ as its
# backend
#######################################################################################################
class PcfgQueue:
    ############################################################################
    # Basic initialization function
    ############################################################################
    def __init__(self):
        self.p_queue = []  ##--The actual priority queue
        self.max_probability = 1.0 #--The current highest priority item in the queue. Used for memory management and restoring sessions
        self.min_probability = 0.0 #--The lowest prioirty item is allowed to be in order to be pushed in the queue. Used for memory management
        self.max_queue_size = 500 #--Used for memory management. The maximum number of items before triming the queue. (Note, the queue can temporarially be larger than this)
        self.reduction_size = self.max_queue_size - self.max_queue_size // 4  #--Target size for the p_queue when it is reduced for memory management
        
        self.storage_list = [] #--Used to store low probability nodes to keep the size of p_queue down
        self.storage_min_probability = 0.0 #-- The lowest probability item allowed into the storage list. Anything lower than this is discarded
        self.storage_size = 10000000 #--The maximum size to save in the storage list before we start discarding items
        
        ##--sanity checks for the data structures for when people edit the above default values
        if self.storage_size < self.max_queue_size:
            raise Exception
        
        
    #############################################################################
    # Push the first value into the priority queue
    # This will likely be 'START' unless you are constructing your PCFG some other way
    #############################################################################
    def initialize(self, pcfg):
        
        ##--Find the START index into the grammar--##
        index = pcfg.start_index()
        if index == -1:
            print("Could not find starting position for the pcfg", file=sys.stderr)
            return RetType.GRAMMAR_ERROR
        
        ##--Push the very first item into the queue--##
        q_item = QueueItem(is_terminal=False, probability = pcfg.find_probability([index,0,[]]), parse_tree = [index,0,[]])
        heapq.heappush(self.p_queue,q_item)
        
        return RetType.STATUS_OK
 
 
    ###############################################################################
    # Memory managment function to reduce the size of the priority queue by
    # deleting the last 1/2 ish of the priority queue
    # It's not an exact number since if multiple items have the same probability
    # and those items fall in the divider of the priority queue then it will save
    # all of them.
    # Aka if the list looks like [0,1,2,3,3,3,7], it will save [0,1,2,3,3,3]
    # If the list looked like [0,1,2,3,4,5,6,7] it will save [0,1,2,3]
    # There is an edge case where no items will be deleted if they all are the same probabilities
    ###############################################################################
    def trim_queue(self):
        ##--First sort the queue so we can easily delete the least probable items--##
        ##--Aka turn it from a heap into a sorted list, since heap pops are somewhat expensive--##
        self.p_queue.sort()
        
        ##--Save the size information about the list
        orig_size = len(self.p_queue)
        
        ##--divider represents the point where we are going to cut the list to remove low probability items
        divider = self.reduction_size
        
        ##--Assign the min probabilty to the item currently in the divider of the queue--##
        self.min_probability = self.p_queue[divider].probability
        print("min prob: " + str(self.min_probability), file=sys.stderr)
        
        ##--Now find the divider we want to cut in case multiple items in the current divider share the same probability
        while (divider < orig_size-1) and (self.p_queue[divider].probability == self.p_queue[divider+1].probability):
            divider = divider + 1
            
        ##--Sanity check for edge case where nothing gets deleted
        if divider == orig_size - 1:
            print("Could not trim the priority queue since at least half the items have the same probability", file=sys.stderr)
            print("Not so much a bug as an edge case I haven't implimented a solution for. Performance is going to be slow until you stop seeing this message --Matt", file=sys.stderr)
        
        ##--Now actually remove the entries--##
        ##--Currently saving them to the storage_list--##
        ##--Need to check to make sure we are not saving items of lower probability then can go into the storage list--##
        storage_end = len(self.p_queue) - 1
        while self.p_queue[storage_end].probability < self.storage_min_probability and storage_end > divider:
            storage_end = storage_end - 1
            
        ##--Copy saved items to the storage_list    
        if storage_end != divider:
            self.storage_list.extend(self.p_queue[divider+1:])
        else:
            print("The 'backup' storage list for memory mangement is getting full. Performance may start to be affected soon", file=sys.stderr)
            
        ##--Delete the entries from the p_queue
        del(self.p_queue[divider+1:])

        ##--Re-heapify the priority queue
        heapq.heapify(self.p_queue)
        
        ##--This can happen if the queue is full of items all of the same probability
        if len(self.p_queue) == orig_size:
            return RetType.QUEUE_FULL_ERROR
        ##--Not an immediate problem but this state will cause issues with resuming sessions. For now report an error state
        if self.min_probability == self.max_probability:
            return RetType.QUEUE_FULL_ERROR
        
        return RetType.STATUS_OK
        
     
    ###############################################################################
    # Used to add items to the priority queue from a previous max probability state
    # End goal is to allow easy rebuilidng and continuation from a previous session
    # This can also be used for memory management so the pqueue can discard nodes that are too
    # low probability as it is running and then rebuild the queue later to bring them back in
    # if the session runs long enough
    #
    # Dev notes: I tried a couple of implimentations of this in the past, but with
    #            the grammar supporting recursion I really struggled with coming up with
    #            and effecient algorithm that was easier than "Run the full session again, (with all the popping, pushing, and
    #            using the next algorithm" until hitting the desired probability threshold
    ###############################################################################
    def rebuild_queue_from_max(self,pcfg):
        print("Functionality to rebuild the priority queue from a previous max probability not implimented yet", file=sys.stderr)
        self.min_probability = 0
        return RetType.STATUS_OK


    ###############################################################################
    # Rebuild the priority queue when it becomes empty
    # Currently just copying items from the storage list back into the priorty queue
    ###############################################################################
    def rebuild_queue(self,pcfg):
        ##--Remove the min probability
        ##--Depending on what type of memory management functionality is in place this may be raised
        ##--at a later point as items get copied back into the priority queue
        self.min_probability = 0
        self.p_queue = []
        
        ##--If there are no items in the storage list--##
        if len(self.storage_list) == 0:
            returnrebuild_queue_from_max(self,pcfg)
            
        ##--Sort the storage list so only the top items go into the priority queue
        self.storage_list.sort()
        
        ##--If we can copy the entire storage list into the priority queue
        if len(self.storage_list) <= self.reduction_size:
            self.p_queue.extend(self.storage_list)
            self.storage_list = []
            self.min_probability = self.storage_min_probability
            
        else:
            divider = self.reduction_size
            self.min_probability = self.storage_list[divider].probability
            while (divider < len(self.storage_list)-1) and (self.storage_list[divider].probability == self.storage_list[divider + 1].probability):
                divider = divider + 1
            
            self.p_queue.extend(self.storage_list[:divider+1])
            del(self.storage_list[:divider+1])
            
        #--Now re-hepify the priority_queue
        heapq.heapify(self.p_queue)
        
        ##--This can happen if the queue is full of items all of the same probability
        if len(self.p_queue) >= self.max_queue_size:
            return RetType.QUEUE_FULL_ERROR
        ##--Not an immediate problem but this state will cause issues with resuming sessions. For now report an error state
        if self.min_probability == self.max_probability:
            return RetType.QUEUE_FULL_ERROR
        
        return RetType.STATUS_OK   
    
    
    #####################################################################################################################
    # Stores a QueueItem in the backup storage mechanism, or drops it depending on how that storage mechanism handles it
    #####################################################################################################################
    def insert_into_backup_storage(self,queue_item):
        if queue_item.probability >= self.storage_min_probability:
            self.storage_list.append(queue_item)
    
        return RetType.STATUS_OK
    
    
    ###############################################################################
    # Pops the top value off the queue and then inserts any children of that node
    # back in the queue
    ###############################################################################
    def next_function(self,pcfg, queue_item_list = []):
        
        ##--Only return terminal structures. Don't need to return parse trees that don't actually generate guesses 
        while True:
            ##--First check if the queue is empty
            while len(self.p_queue) == 0:
                ##--If there was some memory management going on, try to rebuild the queue
                if self.min_probability != 0.0:
                    self.rebuild_queue(pcfg)
                ##--The grammar has been exhaused, exit---##
                else:
                    return RetType.QUEUE_EMPTY
                
            ##--Pop the top value off the stack
            queue_item = heapq.heappop(self.p_queue)
            self.max_probability = queue_item.probability
            ##--Push the children back on the stack
            ##--Currently using the deadbeat dad algorithm as described in my dissertation
            ##--http://diginole.lib.fsu.edu/cgi/viewcontent.cgi?article=5135
            self.add_children_to_queue(pcfg, queue_item)
            
            ##--Memory management
            if len(self.p_queue) > self.max_queue_size:
                print("trimming Queue", file=sys.stderr)
                self.trim_queue()
                print("done", file=sys.stderr)
            ##--If it is a terminal structure break and return it
            if queue_item.is_terminal == True:
                queue_item_list.append(queue_item)
                break

        #print("--Returning this value")
        #print(queue_item_list[0].detailed_print(pcfg), file=sys.stderr)
        return RetType.STATUS_OK

        
    #################################################################################################################################################
    # Adds children to the priority queue
    # Currently using the deadbeat dad algorithm to determine which children to add
    # The deadbead dad "next" algorithm as described in http://diginole.lib.fsu.edu/cgi/viewcontent.cgi?article=5135
    ##################################################################################################################################################
    def add_children_to_queue(self,pcfg, queue_item):
        
        my_children_list = pcfg.deadbeat_dad(queue_item.parse_tree)

        ##--Create the actual QueueItem for each child and insert it in the Priority Queue
        for child in my_children_list:
            child_node = QueueItem(is_terminal = pcfg.find_is_terminal(child), probability = pcfg.find_probability(child), parse_tree = child)
            if child_node.probability <= queue_item.probability:
                ##--Memory management check---------
                ##--If the probability of the child node is too low don't bother to insert it in the queue
                if child_node.probability >= self.min_probability:
                    heapq.heappush(self.p_queue,child_node)
                ##--Else insert it into the backup storage
                else:
                    self.insert_into_backup_storage(child_node)
            else:
                print("Hmmm, trying to push a parent and not a child on the list", file=sys.stderr)

            
        