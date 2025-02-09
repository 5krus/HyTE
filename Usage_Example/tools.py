# Prepare imports.
import pandas as pd
import RapidUseML as RuM # If failing here, run "pip install RapidUseML" in terminal / CMD.

class Tools:

    # Prepare tool schemas.
    # WHY: The models need to know what tools they have and how they tools work.
    TOOL_SCHEMA = {
        "name": "evaluate_design",
        "description": (
            "Given the inputs phi_d, j_d, df and j, return the values of efficiency eta_poly, "
            "Cptt (psi) and phi_op."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "phi_d": {
                    "type": "number",
                    "description": "Design flow coefficient."
                },
                "j_d": {
                    "type": "number",
                    "description": "Advance ratio at design point."
                },
                "df": {
                    "type": "number",
                    "description": "Diffusion factor."
                },
                "j": {
                    "type": "number",
                    "description": "Advance ratio at current operating conditions."
                }
            },
            "required": ["phi_d", "j_d", "df", "j"]
        }
    }

    @staticmethod
    def evaluate_design(phi_d: float, j_d: float, df: float, j: float) -> pd.DataFrame:
        """
        Evaluates the design by predicting eta_poly, phi_op, and Cptt based on input parameters.
        WHY: LLMs uses this to evaluate designs' performances, thereby evaluating their hypotheses.

        Parameters
        ----------
        phi_d : Design flow coefficient.
        j_d : Design advance ratio.
        df : Diffusion factor.
        j : Advance ratio at current operating conditions.

        Returns
        -------
        DataFrame : Input and output pairs, with perfrmance predictions completed by CFD-MLmodels.
        """

        # Create input DataFrame.
        input_data = {
            'phi_d': [phi_d],
            'j_d': [j_d],
            'df': [df],
            'j': [j]
        }
        input_df = pd.DataFrame(input_data)

        # Create an instance of CFD-ML modelling class.
        ml = RuM.ML()

        # Get and return predictions.
        predictions = {}
        for target_column_name in ["eta_poly", "phi_op", "Cptt"]:
            predictions[target_column_name] = ml.predict(input_df, target_column_name)
        return pd.DataFrame(predictions)
