# -*- coding: utf-8 -*-
"""
Classes to calculate and return ETA, make output less messy in automated
processing notebooks.
"""

import time

from datetime import timedelta
from IPython.display import clear_output

__all__ = ['EtaTracker',
           'OutputTracker']

class EtaTracker:
    """A class to calculate an avg. rate and ETA during automated processing.
       A string with the current ETA can be retrieved from self.str_eta.

    Parameters
    ----------
    total_n : int
        Total number of samples that will be run (used to calculate rate
        and ETA).

    """

    def __init__(self, total_n):
        """Initialize eta tracker instance.

        Parameters
        ----------
        total_n : int
            Total number of samples that will be run (used to calculate rate
            and ETA).

        Returns
        -------
        None.

        """
        self.total_n = total_n
        self.curr_rate = 0
        self.curr_n = 0
        self.curr_eta = 0
        self.start_time = None
        self.n_time = None
        self.str_eta = 'ETA: calculating'

    def start(self):
        """Start the timer for a single spot/image.

        Returns
        -------
        None.

        """
        self.n_time = None
        self.start_time = time.perf_counter()

    def stop_update_eta(self):
        """Stop the timer (i.e., spot is done) and update the rate, ETA.

        Returns
        -------
        None.

        """
        self.n_time = time.perf_counter() - self.start_time
        self.adj_eta(self.n_time)
        self.start_time = None


    def adj_eta(self, input_val):
        """Recalculate avg. rate, ETA, updates self.eta_str. Called internally.

        Parameters
        ----------
        input_val : float
            A new spot time in s.

        Returns
        -------
        None.

        """
        self.curr_n += 1
        self.curr_rate = self.curr_rate*(1-(1/self.curr_n)) + input_val*(1/self.curr_n)
        self.curr_eta = (self.total_n - self.curr_n) * self.curr_rate
        if self.curr_n >= 10:
            self.str_eta = 'ETA: '+'{:0>8}'.format(str(timedelta(seconds=round(self.curr_eta, 0))))


class OutputTracker:
    """Class for displaying and/or refreshing text outputs in a Notebook.
       Used to keep outputs in a constant position without constant auto-
       resizing of output box (as is the case when outputs are cleared)
       normally.

    Parameters
    ----------
    n_blank_lines : int, optional
        Expected max lines of output text; blank placeholder lines
        will be used to keep output box size constant. The default is 10.
    stream_outputs : Bool, optional
        If true, all txt inputs to the tracker will be printed normally,
        without clearing/refreshing. The default is False.

    """

    def __init__(self, n_blank_lines=10, stream_outputs=False):
        """Initialize class instance.

        Parameters
        ----------
        n_blank_lines : int, optional
            Expected max lines of output text; blank placeholder lines
            will be used to keep output box size constant. The default is 10.
        stream_outputs : Bool, optional
            If true, all txt inputs to the tracker will be printed normally,
            without clearing/refreshing. The default is False.

        Returns
        -------
        None.

        """
        self.n_blank_lines = n_blank_lines
        self.stream_outputs = stream_outputs
        self.curr_new_txt = []
        self.curr_output = ['\n' for _ in range(n_blank_lines)]

    def print_txt(self, new_txt):
        """Print something using the tracker. This will print normally if
           self.stream_outputs is True.

        Parameters
        ----------
        new_txt : any type
            Something to print; will be converted to string.

        Returns
        -------
        None.

        """
        self.curr_new_txt.append(str(new_txt))

        #make sure that we don't exceed cur output index
        if not self.stream_outputs:
            if len(self.curr_new_txt) > len(self.curr_output):
                for _ in range(len(self.curr_new_txt) - len(self.curr_output)):
                    self.curr_output.append('\n')
                self.n_blank_lines = len(self.curr_new_txt)

            #set curr output block to include new text
            for idx, txt in enumerate(self.curr_new_txt):
                self.curr_output[idx] = txt

            #clear output
            clear_output(wait=True)
            #add print output block
            indiv_output=[string + '\n' if not string.endswith('\n')
                          else string for string in self.curr_output]
            print(''.join(indiv_output), flush=True)
        #if streaming on, just print inputs
        else:
            print(new_txt)

    def reset(self):
        """Reset text in the tracker. This will not do anything if
           self.stream_outputs is True.

        Returns
        -------
        None.

        """
        self.curr_new_txt = []
        self.curr_output = ['\n' for _ in range(self.n_blank_lines)]
        #print blank block
        if not self.stream_outputs:
            #clear output
            clear_output(wait=True)
            #add print output block
            indiv_output=[string + '\n' if not string.endswith('\n')
                          else string for string in self.curr_output]
            print(''.join(indiv_output), flush=True)
