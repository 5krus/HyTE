# Prepare imports.
import os                          # Allows for accessing local system (e.g. to save logs).
import json                        # Allows for simple handling of tool schemas for LLM use.
import openai                      # Allows for OpenAI model access with keys via their API.
import datetime                    # Allows for making timestamps used for live run logging.
from rich import print as rprint   # Allows for custom (colourful) text printing in terminal.

class Iterator:

    def __init__(self):
        self.full_log = ""
        self.iteration_summaries = []
        self.print_enabled = True
        self.logging_enabled = True
        self.logging_folder = "logs"
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self, system_prompts: list, options: dict, key: str, tools, sample_data: any) -> list:
        """
        Orchestrates the hypothesize–test–evaluate workflow.
        WHY: Provides a simple, single-function iterface so non-software engineers can still use it.

        Parameters
        ----------
        system_prompts : Prompt strings defining LLM's roles, tasks, options, etc.
        options : Current run settings like iteration count, model names, logging preferences, etc.
        key : OpenAI API key such that LLMs can be accessed.
        tools : The tools available to the LLMs for testing hypotheses via experiments.
        sample_data : (OPTIONAL) Data to provide the model with as an example of what to expect.

        Returns
        -------
        [summary, full_log]: Summary of iteration discoveries, as well as a full log of everything.
        """

        # Initialise starting values as provided by user.
        iterations = options.get("iterations", 5)
        self.print_enabled = options.get("print-process", True)
        self.logging_enabled = options["logging"].get("log", True)
        self.logging_folder = options["logging"].get("folder", True)
        openai.api_key = key or (rprint("[red]Error loading OpenAI key.[/red]") or None)
        if self.logging_enabled: self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialise history as empty.
        # WHY: These will be used to pass context to subsquent iterations but no history exists yet.
        context, log, previous_feedback = "", "", ""

        for it in range(1, iterations + 1):
            self._print(f"\n[bold cyan]--- Iteration {it} ---[/bold cyan]")
            log += f"\n--- Iteration {it} ---\n"
            if self.logging_enabled: self._save_log(log)

            # For the first iteration, load initial data and include it in the context.
            # WHY: On the first iteration, I sometimes want to provide actual data for more context.
            if it == 1:
                try:
                    if sample_data:
                        context = "Data: " + sample_data.to_dict().__str__()

                except Exception as e:
                    error_message = f"[red]Failed to load initial data: {e}[/red]"
                    if self.print_enabled: self._print(error_message)
                    raise
            else:
                # In subsequent iterations, pass the previous experiment table and evaluation.
                # WHY: This contributes to self-correction / discovery based on prior tested ideas.
                context = "Previous iteration evaluation:\n" + previous_feedback


            ## HYPOTHESIZE.

            # Generate a (new or refined) hypothesis.
            # WHY: There needs to be an idea to test and evaluate. i.e. standard scientific method.
            hypothesis_response = self.generate_hypothesis(
                system_prompt=system_prompts[0],
                context=context,
                model=options["models"]["hypothesizer"]
            )
            self._print(f"[cyan]Hypothesis Response:\n{hypothesis_response}[/cyan]")
            log += f"Iteration {it} Hypothesis Response:\n{hypothesis_response}\n"
            if self.logging_enabled: self._save_log(log)

            # Split the hypothesis response into the hypothesis (+details) and the experiment table.
            # WHY: Makes it simpler for dumber LLMs to process experiment data. Also, cleaner logs.
            hypothesis_text, experiments = self.split_hypothesis_response(hypothesis_response)
            self._print(f"[cyan]Extracted Experiment Table:\n{experiments}[/cyan]")
            log += f"Iteration {it} Extracted Experiment Table:\n{experiments}\n"
            if self.logging_enabled: self._save_log(log)


            ## EXPERIMENT.

            # Run experiments until the experiment table is complete.
            # WHY: Ensures all experiments desired by the hypothesizer are done before progessing.
            experiments = self.run_experiments(experiments, system_prompts, tools,
                experimenter_model=options["models"]["experimenter"],
                completion_checker_model=options["models"]["snitch"],
                max_tokens=4096
            )
            self._print(f"[blue]Experiment Table:\n{experiments}[/blue]")
            log += f"Iteration {it} Experiment Table:\n{experiments}\n"
            if self.logging_enabled: self._save_log(log)


            ## EVALUATE.

            # Evaluate the hypothesis.
            # WHY: This allows for self-correction by checking whether the hypothesis was correct.
            evaluation = self.evaluate_hypothesis(system_prompts[1], experiments, hypothesis_text,
                                                  model=options["models"]["evaluator"]
                                                 )
            self._print(f"[magenta]Evaluation:\n{evaluation}[/magenta]")
            log += f"Iteration {it} Evaluation:\n{evaluation}\n"
            if self.logging_enabled: self._save_log(log)


            # COMPILE ITERATION.

            # Save feedback for the next iteration.
            # WHY: Future iteraitons must know what occured to avoid repeating ideas / experiments.
            iteration_summary = (f"Iteration {it}:\n\n"
                                 f"Hypothesis:\n{hypothesis_text}\n\n"
                                 f"Experiments:\n{experiments}\n\n"
                                 f"Evaluation:\n{evaluation}")
            self.iteration_summaries.append(iteration_summary)

        # Return logs and summaries for user to play with.
        # WHY: While only local saves could work, having these as a response makes recursion easier.
        self.full_log, summary = log, "\n".join(self.iteration_summaries)
        return [summary, log]


    ### MAIN H-T-E FUNCTONS ###

    def generate_hypothesis(self, system_prompt: str, context: str, model: str,
                            max_tokens: int = 4096) -> str:
        """
        Generates a hypothesis (initial or refined) using the provided system prompt and context.
        WHY: There needs to be an idea to test and evaluate. i.e. standard scientific method.

        Parameters
        ----------
        system_prompt : Prompt defining LLM's role, task, option, etc.
        context : Prior hypotheses, experiments conducted, and evaluations, if any were conducted.
        model : The name of the model to make the query with. i.e. "4o", "o1", "o3-mini-high", etc.
        max_tokens : Limit on number of words in response. 1 token ~4/5 of a word => 2kt ~= 1.6kw.

        Returns
        -------
        response : LLM's reponse. i.e. It's hypothesis, reasoning, desired experiments, etc.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context}
        ]
        response = self.llm_call(model, messages, max_tokens)
        return response.strip()

    def run_experiments(self, experiments: str, system_prompts: list, tools,
                        experimenter_model: str, completion_checker_model: str,
                        max_tokens: int = 4096) -> str:
        """
        Runs experiments until the specified requirements are complete.
        WHY: Having completed experiments allows for the validation / disproving of hypothesis.

        Note:
        - system_prompts[1] used as the Experiment Runner prompt, with functions enabled.
        - system_prompts[4] used as the Experiment Completion Checker prompt, with no functons.

        Parameters
        ----------
        experiments : String containing experiments specified by LLM hypothersizer.
        system_prompts : Prompt strings defining LLM's roles, tasks, options, etc.
        tools : Class of tool functions for running experiments, provided alognside their schema.
        experimenter_model : Name of the model to be used for conducting experiments.
        completion_checker_model : Name of the model to be used for checking experiment completion.
        max_tokens : Limit on number of words in response. 1 token ~4/5 of a word => 2kt ~= 1.6kw.

        Returns
        -------
        experiments : Completed experiments.
        """
        attempts = 0
        while attempts < 5:
            # Call Experiment Runner with previously specified experiments.
            messages = [
                {"role": "system", "content": system_prompts[1]},
                {"role": "user", "content": experiments}
            ]
            response = self.llm_call(experimenter_model, messages, max_tokens,
                                     functions=tools.TOOL_SCHEMAS)
            message = response.choices[0].message

            if message.get("function_call"):
                # If the LLM wants to call a function, extract relevant details and run it.
                function_name = message.function_call.name
                try:
                    arguments = json.loads(message.function_call.arguments)
                except Exception as e:
                    error_message = f"[red]Error parsing function arguments: {e}[/red]"
                    if self.print_enabled: self._print(error_message)
                    break

                # FUTURE ERYK: MAKE THIS SHIT LESS SHIT PELASE OH MY GOD.
                if function_name == "evaluate_design":
                    # Execute the tool function.
                    result_df = tools.evaluate_design(
                        arguments['phi_d'],
                        arguments['j_d'],
                        arguments['df'],
                        arguments['j']
                    )
                    # Send the function result back to the model.
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "function_call": message.function_call
                    })
                    messages.append({
                        "role": "function",
                        "name": function_name,
                        "content": result_df.to_json()
                    })
                    response2 = self.llm_call(experimenter_model, messages, max_tokens, functions=tools.TOOL_SCHEMAS)
                    experiments = response2.choices[0].message.content
            else:
                # No function call in the response: check if experiments are complete.
                messages2 = [
                    {"role": "system", "content": system_prompts[4]},
                    {"role": "user", "content": experiments}
                ]
                response3 = self.llm_call(completion_checker_model, messages2, max_tokens)
                checker_reply = response3.choices[0].message.content.upper()
                if "COMPLETE" in checker_reply:
                    break
                else:
                    # Update the experiment table with the model's new output.
                    experiments = response3.choices[0].message.content

        # Increment attempts to avoid infinite loops if LLMs fail.
        attempts += 1
        return experiments

    def evaluate_hypothesis(self, system_prompt: str, experiments: str, hypothesis_text: str,
                            model: str, max_tokens: int = 4096) -> json:
        """
        Evaluates the hypothesis by sending the hypothesis text and the complete experiment table
        to the LLM evaluator.
        WHY: To know whether the hypothesis was right, it needs to be checked against expeirments.

        Parameters
        ----------
        system_prompt : Prompt defining LLM's role, task, option, etc.
        experiments : String containing completed experiments.
        hypothesis_text : String containing original hypothesis to be evaluated by LLM here.
        model : The name of the model to make the query with. i.e. "4o", "o1", "o3-mini-high", etc.
        max_tokens : Limit on number of words in response. 1 token ~4/5 of a word => 2kt ~= 1.6kw.

        Returns
        -------
        response : Evaluation of hypothesis correctness based on experimental results.
        """
        user_message = f"Hypothesis:\n{hypothesis_text}\n\nExperiment Results:\n{experiments}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        response = self.llm_call(model, messages, max_tokens)
        return response.strip()


    ### SUPPORT FUNCTIONS ###

    def _print(self, message: str) -> None:
        """
        Prints to terminal only if printing is enabled.
        WHY: To avoid spamming terminal if the user prefers it to be clean.

        Parameters
        ----------
        message : Text to be printed if printing is enabled.

        Returns
        -------
        None
        """
        if self.print_enabled:
            rprint(message)

    def _save_log(self, log: str) -> None:
        """
        Saves timestampted log to a file within 'logs' folder. Creates folder if it doesn't exist.
        WHY: Allows for the ability to revisit past runs, if the user wants to save them.

        Parameters
        ----------
        log : Iteration log (progress of run) to be saved.

        Returns
        -------
        None
        """
        if not os.path.exists(self.logging_folder): os.makedirs(self.logging_folder)
        log_filename = os.path.join(self.logging_folder, f"run_{self.timestamp}.txt")

        # Updates contents of log file, named with timestamp created at run start.
        # WHY: Files are timestamped to avoid overwriting existing logs.
        try:
            with open(log_filename, "w", encoding="utf-8") as f:
                f.write(log)
        except Exception as e:
            error_message = f"[red]Failed to save log: {e}[/red]"
            if self.print_enabled: self._print(error_message)


    def llm_call(self, model: str, messages: list, max_tokens: int = 4096,
                 functions = None) -> json:
        """
        Calls the OpenAI ChatCompletion API with the given messages.
        WHY: Using OpenAI's models avoids us having to build an LLM from scratch. i.e. Cost savings.

        Parameters
        ----------
        model : The name of the model to make the query with. i.e. "4o", "o1", "o3-mini-high", etc.
        messages : List of "prior messages from chat", as well as prompt for LLM to respond to.
        max_tokens : Limit on number of words in response. 1 token ~4/5 of a word => 2kt ~= 1.6kw.
        functions : Details of functions available to the LLM such that it can perform expeirments.

        Returns
        -------
        response : LLM's reponse. This depends on context and your request.
        """
        if functions:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    functions=functions
                )
                return response.choices[0].message
            except Exception as e:
                error_message = f"[red]LLM call failed: {e}[/red]"
                if self.print_enabled: self._print(error_message)
                raise
        else:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                error_message = f"[red]LLM call failed: {e}[/red]"
                if self.print_enabled: self._print(error_message)
                raise

    def split_hypothesis_response(self, response_text: str):
        """
        Uses an LLM call to robustly parse the hypothesis response into two parts:
        - hypothesis_text: the textual description of the hypothesis and reasoning.
        - experiment_table: the block of text containing experiment definitions.
        WHY: Using an LLM for this instead of RegEx allows for generalisation.

        Parameters
        ----------
        response_text : String from LLM containing hypothesis text and desired experiments list.

        Returns
        -------
        [hypothesis_text, experiment_table] : String containing only written hypothesis, and string
                                              containing only experiment data.
        """
        # Define a system prompt that instructs the LLM to perform the splitting.
        system_prompt = (
        "You are an expert information parser. It is your job is to extract two distinct parts "
        "from the provided text. The first part is the person's hypothesis (and related details)."
        "Details like the idea, reasoning behind, desired outcomes, and also other such things."
        "The second part is experimental data, which may come in weird looking formats but it will "
        "be notably distinct from the hypothesis, reasoning and other 'readable' text. Hopefully, "
        "if the prior worker did their job correctly, the experiment section will be labelled, "
        "which should make things much easier for you. Return your answer in valid JSON format "
        "with exactly two keys: 'hypothesis_text' and 'experiment_data'. The contents of those two "
        "keys should be strings. If no experiment table is present, return an empty string for "
        "'experiment_data'."
        ) # Hard-coding this prompt because 99% chance nobody will touch it anyway.

        # Prepare the user prompt with the response text.
        user_prompt = f"Input text:\n{response_text}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call the LLM using the built-in llm_call method.
        # Model is hard-coded to "4o" because 99% chance nobody will want to change this.
        try:
            # Extract hypothesis and experiment data as separate JSONs.
            llm_output = self.llm_call(model="4o", messages=messages, max_tokens=4096)
            parsed_output = json.loads(llm_output)
            hypothesis_text = parsed_output.get("hypothesis_text", "").strip()
            experiment_table = parsed_output.get("experiment_table", "").strip()
            return hypothesis_text, experiment_table

        except Exception as e:
            raise Exception(f"LLM-based parsing failed: {e}")