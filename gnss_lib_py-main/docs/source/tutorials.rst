.. _tutorials:

Tutorials
=========


How to Load Android Derived Dataset
-----------------------------------

This example shows how to quickly load in a Android derived dataset
file into the :code:`gnss_lib_py` framework.

.. code-block:: python

    from gnss_lib_py.parsers.android import AndroidDerived
    derived = AndroidDerived(derived_dataset_path)

How to Create a New Measurement Child Class
-------------------------------------------
The modular and versatile functionality of this :code:`gnss_lib_py`
repository is enabled by loading all measurement data types into a
custom Python `Measurement class <https://github.com/Stanford-NavLab/gnss_lib_py/blob/main/gnss_lib_py/parsers/measurement.py>`__.
If your measurements use a file type not already supported in the list
on our :ref:`main page<mainpage>`, then you will need to create a new
child Measurement Python class. This tutorial will guide you on how to
set up your custom Python class. Once complete, please feel free to
submit a pull request to our GitHub repository so other users can also
make use of the added functionality.

1. Create preprocess.

2. Create postprocess. Must be defined for a valid child class. Use 
   :code:`pass` inside the function definition if not performing any 
   operations.

3. Create a new :code:`_column_map()` function that translates the
   column names from the new measurement type into our standard names.

Your finished class might look something like:

.. code-block:: python

    from gnss_lib_py.parsers.measurement import Measurement

    class NewMeasurementType(Measurement):
    """Class handling derived measurements from Android dataset.
    Inherits from Measurement().
    """
    #NOTE: Inherits __init__() and isn't defined explicitly here because
    # no additional implementations/attributes are needed

    def preprocess(self, input_path):
        """Loading and preprocessing.

        Parameters
        ----------
        input_path : string
            Path to measurement csv file
        Returns
        -------
        pd_df : pd.DataFrame
            Loaded measurements with consistent column names
        """
        pd_df = pd.read_csv(input_path)
        col_map = self._column_map()
        pd_df.rename(columns=col_map, inplace=True)
        return pd_df

    def postprocess(self):
        """Postprocessing.

        """
        pass

    @staticmethod
    def _column_map():
        """Map of column names from loaded to gnss_lib_py standard

        Returns
        -------
        col_map : Dict
            Dictionary of the form {old_name : new_name}
        """
        col_map = {'millisSinceGpsEpoch' : 'toeMillis',
                'svid' : 'PRN',
                'xSatPosM' : 'x_sat_m',
                'ySatPosM' : 'y_sat_m',
                'zSatPosM' : 'z_sat_m',
                'xSatVelMps' : 'vx_sat_mps',
                'ySatVelMps' : 'vy_sat_mps',
                'zSatVelMps' : 'vz_sat_mps',
                'satClkBiasM' : 'b_sat_m',
                'satClkDriftMps' : 'b_dot_sat_mps',
                }
        return col_map



Visualize your Data
-------------------

Examples of how to visualize data.
