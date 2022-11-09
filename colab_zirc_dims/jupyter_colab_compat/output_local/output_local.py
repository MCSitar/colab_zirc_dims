# -*- coding: utf-8 -*-
"""
Hacky, minimal reproduction of the google.colab package .output module 
for local IPython >= 7.0 execution of GUI using the same calls
as from the .output module (e.g., redirect_to_element,
register_callback, etc.)
"""

import json
import contextlib
import ipywidgets as widgets
from IPython.display import display, clear_output, Javascript

__all__ = ['invoke_function', 'register_callback',
           'display_outputs', 'clear', 
           'redirect_to_element', 'alt_eval_js']

_json_decoder = json.JSONDecoder()


def invoke_function(function_name, json_args, json_kwargs):
    """Function modified from google.colab; see license string
    in module __init__.py file.
    
    Invokes callback with given function_name.

    This function is meant to be used by frontend when proxying
    data from secure iframe into kernel.  For example:

    invoke_function(fn_name, "'''"   + JSON.stringify(data) + "'''")

    Note the triple quotes: valid JSON cannot contain triple quotes,
    so this is a valid literal.

    Args:
    function_name: string
    json_args: string containing valid json, provided by user.
    json_kwargs: string containing valid json, provided by user.

    Returns:
    The value returned by the callback.

    Raises:
    ValueError: if the registered function cannot be found.
    """
    #with GUI_out_widget:
    #    print('Debug: alt_invoke fxn running')
    args = _json_decoder.decode(json_args)
    kwargs = _json_decoder.decode(json_kwargs)

    callback = _functions.get(function_name, None)
    if not callback:
        raise ValueError('Function not found: {function_name}'.format(
            function_name=function_name))

    return callback(*args, **kwargs)

#dict for functions registered using register_callback
_functions = {}


def register_callback(function_name, callback):
    """Function modified from google.colab; see license string
    in module __init__.py file.
    
    Registers a function as a target invokable by Javacript in outputs.

    This exposes the Python function as a target which may be invoked by
    Javascript executing in Colab output frames.

    This callback can be called from javascript side using:
    colab.kernel.invokeFunction(function_name, [1, 2, 3], {'hi':'bye'})
    then it will invoke callback(1, 2, 3, hi="bye")

    Args:
    function_name: string
    callback: function that possibly takes positional and keyword arguments
    that will be passed via invokeFunction()
    """
    _functions[function_name] = callback


#output widgets for a GUI
GUI_out_widget = widgets.Output(layout={'border': '1px solid black'})
addit_print_out= widgets.Output()

def display_outputs():
    """Clear and display output widgets for GUI.

    Returns
    -------
    None.

    """
    GUI_out_widget.clear_output(False)
    addit_print_out.clear_output(False)
    clear_output()
    display(GUI_out_widget)
    display(addit_print_out)

def clear():
    """Clear and display outputs. Wrapper for display_outputs().

    Returns
    -------
    None.

    """
    display_outputs()

@contextlib.contextmanager
def redirect_to_element(*args):
    """A contextmanager for an additional output widget. Workaround
    for buggy callback injections of JS code to change js elements 
    in the GUI itself.

    Parameters
    ----------
    *args : Any
        These do not do anything. Provided for compatibility with
        google.colab.output.redirect_to_element function.

    Returns
    -------
    None.

    """
    addit_print_out.clear_output(False)
    with addit_print_out:
        yield

def alt_eval_js(inpt_async_js_fxn, inpt_call_fxn_str):
    """Crude equivalent to google.colab.output.eval_js(); simply appends
       a string to a Ipython.display.Javascript object then displays
       both as a single js object.

    Parameters
    ----------
    inpt_async_js_fxn : Ipython.display.Javascript instance
        A javascript object defining an async function.
    inpt_call_fxn_str : str
        A string with additional js code, args to run the previously
        defined async function. For example:
            ("run_my_async_fxn(arg1, arg2...)
        Raw args must be js-interpretable.

    Returns
    -------
    None.

    """
    full_js_to_run = inpt_async_js_fxn.data + inpt_call_fxn_str
    with GUI_out_widget:
        display(Javascript(full_js_to_run))