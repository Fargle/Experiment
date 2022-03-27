from dataclasses import dataclass
from gc import callbacks

import pandas as pd
import streamlit as st


def write_to_screen(results):
    """Write results of Selection to Screen"""
    st.write(results)


def add_selectbox(name: str, items: tuple):
    """Add Selection Box to Sidebar"""
    selectbox = st.sidebar.selectbox(name, items)
    return selectbox


@dataclass
class App:
    """Defines an Analysis Application"""

    callbacks: dict

    def build(self):
        """build the GUI"""
        add_selectbox("Model Analysis", tuple(self.callbacks.keys()))
        self.selections["Feature"] = add_selectbox(
            "Feature Analysis", tuple(self.callbacks["FeatureAnalysis"].keys())
        )

    def run(self):
        """Run the Application"""

        write_to_screen(self.callbacks[self.selections["Model"]]())
