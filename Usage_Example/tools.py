"""
Tool Usage Example.

This script shows how function-based tools can be made accessible to LLMs.
"""

# Prepare imports.
import warnings
import pandas as pd
import RapidUseML as RuM  # If failing here, run "pip install RapidUseML" in Terminal / CMD.
from sklearn.exceptions import InconsistentVersionWarning

class Tools: # pylint: disable=too-few-public-methods.
    """
    This class defines functions that are used as tools by LLMs, as well as their tool schemas
    describing how the tools are used.
    """

    # Prepare tool schemas.
    # WHY: The models need to know what tools they have and how they tools work.
    TOOL_SCHEMA = [{
        "name": "evaluate_design",
        "description": (
            "Given an array of experiments, where each experiment is represented as an array of "
            "four numbers [phi_d, j_d, df, j], return an array of predictions. Each prediction is "
            "an object containing the keys: eta_poly, phi_op, and Cptt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "experiments": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "A single experiment with values: [phi_d, j_d, df, j]."
                    }
                }
            },
            "required": ["experiments"]
        }
    }]

    @staticmethod
    def evaluate_design(experiments: list) -> pd.DataFrame:
        """
        Evaluates the design by predicting eta_poly, phi_op, and Cptt based on input parameters.
        WHY: LLMs uses this to evaluate designs' performances, thereby evaluating their hypotheses.

        Parameters
        ----------
        experiments : list of lists containing values of experiments to be completed.

        Returns
        -------
        DataFrame : Input and output pairs, with perfrmance predictions completed by CFD-MLmodels.
        """

        ## Suppressing warnings from RapidUseML to have a clean terminal. Issue is version related.
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

        # Convert the list of experiment dicts to a DataFrame.
        input_df = pd.DataFrame(experiments, columns=["phi_d", "j_d", "df", "j"])

        # Create an instance of CFD-ML modelling class.
        ml = RuM.ML()  # pylint: disable=no-member.

        # Get and return predictions.
        predictions = {}
        for target_column_name in ["eta_poly", "phi_op", "Cptt"]:
            predictions[target_column_name] = ml.predict(input_df, target_column_name)

        # Convert predictions dict to a DataFrame.
        pred_df = pd.DataFrame(predictions)

        # Combine inputs and outputs into one DataFrame.
        # FUTURE ERYK. NEED TO VALIDATE THAT THIS "FREEFLOW" METHOD ACTUALLY MATCHES ROWS CORRECTLY.
        combined_df = pd.concat([input_df.reset_index(drop=True),
                                 pred_df.reset_index(drop=True)], axis=1)
        return combined_df
