from dataclasses import dataclass, field
from gc import callbacks
from typing import List

import pandas as pd
import streamlit as st
from attr import field


@st.cache
def write_to_screen(results):
    """Write results of Selection to Screen"""
    st.write(results)


def add_selectbox(name: str, options):
    """Add Selection Box to Sidebar"""
    selectbox = st.sidebar.selectbox(label=name, options=options)
    return selectbox


@dataclass
class App:
    """Defines an Analysis Application"""

    callbacks: List
    callback_lookup = {}
    selections = []

    def build(self):
        """build the GUI"""
        print("hello")
        all_selectboxes = {}

        no_selection = ["None"]
        print("outside loop")
        for callback in self.callbacks:
            print(callback)
            if str(callback) in all_selectboxes:
                all_selectboxes[str(callback)].append(callback.display_name)
            else:
                all_selectboxes[str(callback)] = [callback.display_name]

            self.callback_lookup[callback.display_name] = callback

        for name, selection_choice in all_selectboxes.items():
            print(name)
            self.selections.append(add_selectbox(name, no_selection + selection_choice))

    def run(self):
        """Run the Application"""
        for selection in self.selections:
            if selection is not None:
                write_to_screen(self.callback_lookup[selection].conduct())
