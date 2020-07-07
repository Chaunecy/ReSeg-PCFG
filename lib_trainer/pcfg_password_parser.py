#!/usr/bin/env python3


#############################################################################
# This file contains the functionality to parse raw passwords for PCFGs
#
# The PCFGPasswordParser class is designed to be instantiated once and then
# process one password at at time that is sent to it
#
#############################################################################
import sys
import traceback
from collections import Counter

from .base_structure import base_structure_creation
from .context_sensitive_detection import context_sensitive_detection
# Local imports
from .keyboard_walk import detect_keyboard_walk
from .my_leet_detector import MyL33tDetector
from .prince_metrics import prince_evaluation
from .trainer_file_input import TrainerFileInput
from .year_detection import year_detection


## Responsible for parsing passwords to train a PCFG grammar
#
class PCFGPasswordParser:

    ## Initializes the class and all the data structures
    #
    # multiword_detector: A previously trained multi word detector
    #                      that has had the base_words established for it
    #
    def __init__(self, multiword_detector, save_structs=None):

        # Save the multiword detector
        self.multiword_detector = multiword_detector

        # Initialize Leet Speak Detector
        self.leet_detector = MyL33tDetector(self.multiword_detector)

        # Used for debugging/statistics
        #
        # These numbers won't add up to total passwords parsed since
        # some passwords might have multiple "base words". For example
        # "pass1pass" would be counted as two single words. Likewise,
        # "123456" would have no words
        #
        self.num_single_words = 0
        self.num_multi_words = 0

        # Keep track of the number of leet replacements detected
        self.num_leet = 0

        ## The following counters keep track of global running stats
        #
        self.count_keyboard = {}
        self.count_emails = Counter()
        self.count_email_providers = Counter()
        self.count_website_urls = Counter()
        self.count_website_hosts = Counter()
        self.count_website_prefixes = Counter()
        self.count_years = Counter()
        self.count_context_sensitive = Counter()
        self.count_alpha = {}
        self.count_alpha_masks = {}
        self.count_digits = {}
        self.count_other = {}
        self.count_base_structures = Counter()
        self.count_raw_base_structures = Counter()
        self.count_prince = Counter()
        if save_structs is not None and save_structs.writable():
            self.save2 = save_structs
        else:
            self.save2 = None

    def init_l33t(self, training_set, encoding):
        file_input = TrainerFileInput(training_set, encoding)
        num_parsed_so_far = 0
        try:
            password = file_input.read_password()
            while password:

                # Print status indicator if needed
                num_parsed_so_far += 1
                if num_parsed_so_far % 1000000 == 0:
                    print(str(num_parsed_so_far // 1000000) + ' Million')
                # pcfg_parser.parse(password)
                self.leet_detector.detect_l33t(password)
                # Get the next password
                password = file_input.read_password()

        except Exception as msg:
            traceback.print_exc(file=sys.stdout)
            print("Exception: " + str(msg))
            print("Exiting...")
            return
        print(f"init l33t done", file=sys.stderr)
        pass

    # Main function called to parse an individual password
    #
    # Returns:
    #    True: If everything worked correctly
    #    False: If there was a problem parsing the password
    #
    def parse(self, password):

        # Since keyboard combos can look like many other parsings, filter them
        # out first
        section_list, found_walks = detect_keyboard_walk(password)

        self._update_counter_len_indexed(self.count_keyboard, found_walks)

        # Identify e-mail and web sites before doing other string parsing
        # this is because they can have digits + special characters

        # found_emails, found_providers = email_detection(section_list)
        #
        # for email in found_emails:
        #     self.count_emails[email] += 1
        # for provider in found_providers:
        #     self.count_email_providers[provider] += 1
        #
        # found_urls, found_hosts, found_prefixes = website_detection(section_list)
        #
        # for url in found_urls:
        #     self.count_website_urls[url] += 1
        # for host in found_hosts:
        #     self.count_website_hosts[host] += 1
        # for prefix in found_prefixes:
        #     self.count_website_prefixes[prefix] += 1

        # Identify years in the dataset. This is done before other parsing
        # because parsing after this may classify years as another type

        found_years = year_detection(section_list)

        for year in found_years:
            self.count_years[year] += 1

        # Need to classify context sensitive replacements before doing the
        # straight type classifications, (alpha, digit, etc), but want to doing
        # it after other types of classifations.

        found_context_sensitive_strings = context_sensitive_detection(section_list)

        for cs_string in found_context_sensitive_strings:
            self.count_context_sensitive[cs_string] += 1

        section_list, leet_list, mask_list = self.leet_detector.parse_sections(section_list)
        self._update_counter_len_indexed(self.count_alpha, leet_list)
        self._update_counter_len_indexed(self.count_alpha_masks, mask_list)
        # Identify pure alpha strings in the dataset

        section_list, alpha_list, mask_list, digits_list, specials_list \
            = self.multiword_detector.parse_sections(section_list)
        # found_alpha_strings, found_mask_list = alpha_detection(section_list, self.multiword_detector)
        #
        self._update_counter_len_indexed(self.count_alpha, alpha_list)
        self._update_counter_len_indexed(self.count_alpha_masks, mask_list)
        self._update_counter_len_indexed(self.count_digits, digits_list)
        self._update_counter_len_indexed(self.count_other, specials_list)

        # Identify pure digit strings in the dataset

        # found_digit_strings = self.multiword_detector.parse_sections(section_list)
        #
        # self._update_counter_len_indexed(self.count_digits, found_digit_strings)

        # Categorize everything else as other

        # found_other_strings = self.multiword_detector.parse_sections(section_list)
        #
        # self._update_counter_len_indexed(self.count_other, found_other_strings)

        # Calculate the counts of the individual sections for PRINCE dictionary 
        # creation

        prince_evaluation(self.count_prince, section_list)

        # Now after all the other parsing is done, create the base structures
        if self.save2 is not None:
            write2 = [f"{password}"]
            for sec, tag in section_list:
                write2.append(f"{sec}\t{tag}")
            self.save2.write("\t".join(write2))
            self.save2.write("\n")
        is_supported, base_structure = base_structure_creation(section_list)

        if is_supported:
            self.count_base_structures[base_structure] += 1

        self.count_raw_base_structures[base_structure] += 1

        return True

    # Updates a Python Counter object when the item is lenght indexed
    #
    # For example, if the individual counts are broken up by length of input
    # Aka A1 = 'a', A3 = 'cat', A5 = 'chair'
    #
    # Input Values:
    #
    # self: Since this is a class private function
    #
    # input_counter: The Python Counter object to update
    #
    # input_list: A list of items to update in the counter
    #
    def _update_counter_len_indexed(self, input_counter, input_list):

        # Go through every item in the list to insert it in the counter
        for item in input_list:
            # First try a blind insertion into the list
            try:
                input_counter[len(item)][item] += 1

            # If that length index doesn't exist, it'll throw an exception some
            # now create it
            except:
                input_counter[len(item)] = Counter()
                input_counter[len(item)][item] += 1
