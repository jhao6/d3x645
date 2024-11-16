import random
from collections import defaultdict

def construct_meta_task(dataset, num_tasks=500, samples_per_class=10, support_size=5):
    data_by_class = defaultdict(list)
    for example in dataset:
        label = example["labels"].item()
        data_by_class[label].append(example)

    # Build tasks
    tasks = []

    for _ in range(num_tasks):
        # Randomly select 2 classes for the task
        selected_classes = random.sample(data_by_class.keys(), 2)

        # Randomly sample the specified number of examples from each selected class
        task_data = {"support": [], "query": []}
        for cls in selected_classes:
            if len(data_by_class[cls]) >= samples_per_class:
                # Randomly sample 50 data points for the class
                samples = random.sample(data_by_class[cls], samples_per_class)
                support_set = samples[:support_size]  # First 25 for support
                query_set = samples[support_size:]  # Remaining 25 for query

                task_data["support"].extend(support_set)
                task_data["query"].extend(query_set)

        # Add the task data to the list of tasks
        tasks.append(task_data)
    return tasks