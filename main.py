## Library imports
try:
    import pandas as pd
    import numpy as np
    import time
    import random
    from matplotlib import pyplot as plt
    import os
    from multiprocessing import Pool, Manager    
    import ollama
except ImportError as e:
    raise Exception(f"Error importing libraries: {e}")

## World Class (Which instances are labeled as 'model')

class world:
    def __init__(self, no_workers, no_days):
        self.no_workers = no_workers
        self.no_days = no_days
        self.blue_shirts = 0
        self.worker_list = []
        self.blue_over_time = []

        # Initialize workers
        for i in range(no_workers):
            office_worker = worker(id= i, model= self, name= list_of_names[i], traits= list_of_traits[i])
            if random.random() < 0.5:
                office_worker.clothes = "blue"
                office_worker.memory = np.append(office_worker.memory, 1)
            else:
                office_worker.clothes = "green"
                office_worker.memory = np.append(office_worker.memory, 0)
            self.worker_list.append(office_worker)

    # Function to count blue shirts and update worker's info
    def set_clothes_info(self):
        for worker in self.worker_list:
            if worker.clothes == "blue":
                self.blue_shirts += 1
        self.blue_over_time.append(self.blue_shirts)
        info = self.blue_shirts
        self.blue_shirts = 0
        for worker in self.worker_list:
            worker.cumulative_shirts_info = info
        
    def run(self):
        for i in range(self.no_days + 1):
            self.set_clothes_info()
            print(f"Day {i} - Blue shirts: {self.blue_over_time[-1]}/{self.no_workers}\n\n")
            if i == self.no_days:
                break
            for worker in self.worker_list:
                worker.decide_clothes()



## Worker class

class worker:

    # Constructor
    def __init__(self, id, model, name, traits, clothes = None):
        self.id = id
        self.model = model
        self.name = name
        self.traits = traits
        self.clothes = clothes
        # 1 for blue shirt, 0 for green shirt
        self.memory = np.array([], dtype=int)
        self.cumulative_shirts_info = None
        
    # tested models: llama3.1:8b, gpt-oss:20b, llama3.2:3b
    def get_output_from_LLM(self, prompt, model = "llama3.1:8b"):
        success = False
        retry = 0
        max_retries = 5
        while retry < max_retries and not success:
            try:
                result = ollama.generate(model, prompt)
                success = True
            except Exception as e:
                print(f"Error generating response: {e}\nRetrying")
                retry += 1
                time.sleep(0.5)
        if not success:
            raise Exception("Failed to get a response from the LLM after multiple retries.")
        return result['response']
    
    def decide_clothes(self):
        prompt = f"""
        You are {self.name}. You are a {self.traits} person.
        You work in an office with {self.model.no_workers-1} other people. You want to be successful, and earn money.
        You need to decide between wearing blue shirt or green shirt to work. The weather is appropriate for either color. You like to be comfortable at work.
        You chose to wear {self.clothes} shirt yesterday.
        Out of {self.model.no_workers} employees, yesterday, {self.cumulative_shirts_info} wore blue shirts, and {self.model.no_workers-self.cumulative_shirts_info} wore green shirts at the office.

        Based on the above context, you need to choose whether to wear blue or green shirt.
        You must provide your reasoning for your choice and then your response in one word.
        For example, if your answer is "blue”, your response will be:
        Reasoning: [Your reason to choose to wear blue shirt]
        Response: blue
        If your answer is "green”, then your response will be :
        Reasoning: [Your reason to choose to wear green shirt]
        Response: green
        Please make sure your response is in one word.
        """

        try:
            output = self.get_output_from_LLM(prompt)
        except Exception as e:
            print(f"Error in get_output_from_LLM: {e}")
        
        # Parse the output
        reasoning, response = "", ""
        try:
            content = output.split("Reasoning: ", 1)[1]
            reasoning, response = content.split("Response:")
            response = response.strip().split("." ,1)[0]
            reasoning = reasoning.strip()

        except:
            print(f"Error parsing output: {output}")
            response = "blue"  # Default response if parsing fails
            reasoning = "Parsing error, defaulting to blue shirt."

        print(f"Worker {self.name} decided to wear {response} shirt. Reasoning: {reasoning}")

        # Update clothes and memory based on response
        if response.lower() == "blue":
            self.clothes = "blue"
            self.memory = np.append(self.memory, 1)
        elif response.lower() == "green":
            self.clothes = "green"
            self.memory = np.append(self.memory, 0)
        else:
            print(f"Unexpected response: {response}. Defaulting to blue shirt.")
            self.clothes = "blue"
            self.memory = np.append(self.memory, 1)

        
# single process
def process_iteration(i, stop_event):
    if stop_event.is_set():
        return None

    model = world(no_workers = 20, no_days = 7)
    model.run()
    agent_choices = []
    for w in model.worker_list:
        agent_choices.append(w.memory)

    column_names = [f"Day {i}" for i in range(model.no_days + 1)]
    index_names = list_of_names[:model.no_workers]
    df = pd.DataFrame(agent_choices, index=index_names, columns=column_names)
    root = os.getcwd()
    csv_path = root+ f"/outputs/Agent_Choices_per_Day-run-{i+1}.csv"
    df.to_csv(csv_path)

    return csv_path


# Variables
list_of_names = ["Adrian" ,"Mark" ,"Greg" ,"John" ,"Peter" ,"Liz" ,"Rosa" ,"Patricia" ,"Julia" ,"Kathy",
                 "William","Benjamin","Mike", "David", "George","Emma", "Olivia","Elizabeth","Isabella","Mia"]
'''list_of_names = ["Amina" ,"Layla" ,"Fatima" ,"Nasima" ,"Sara" ,"Leila" ,"Hana" ,"Yasmin" ,"Amira" ,"Rima",
                 "Omar","Youssef","Ali", "Hassan", "Karim","Mariam","Nadia","Salma","Dina","Lina"]'''
list_of_traits = ["extremely conformist","highly conformist", "conformist", "low conformist", "non-conformist",
                  "extremely conformist","highly conformist", "conformist", "low conformist", "non-conformist",
                  "highly conformist","conformist", "conformist", "conformist", "low conformist",
                  "highly conformist","conformist", "conformist", "conformist", "low conformist"]


if __name__ == "__main__":
    try:
        os.mkdir("./outputs")
        num_iterations = 1
        pool = Pool(processes=1)

        with Manager() as manager:
            stop_event = manager.Event()
            results = []

            try:
                results = pool.starmap(process_iteration, [(i, stop_event) for i in range(num_iterations)])
            except KeyboardInterrupt:
                print("Keyboard Interrupt detected. Stopping...")
                stop_event.set()
                pool.terminate()
                exit()
            pool.close()
            pool.join()

            main_df = pd.DataFrame()
            for i, csv_path in enumerate(results):
                if csv_path is not None:
                    sub_df = pd.read_csv(csv_path, index_col=0)
                    sub_df.columns = pd.MultiIndex.from_product([["run-{}".format(i + 1)], sub_df.columns])
                    main_df = pd.concat([main_df, sub_df], axis=1)

            main_df.to_csv("consolidated_results.csv")

    except FileExistsError:
        print("./outputs_default_run already exists. Please remove it from the directory!")