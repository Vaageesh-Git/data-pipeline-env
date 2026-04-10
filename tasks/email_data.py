from tasks import TASKS


TASK_EMAILS = {
    task["id"]: task["files"]
    for task in TASKS
}

TASK_OBJECTIVES = {
    task["id"]: task["description"]
    for task in TASKS
}

TASK_MAX_STEPS = {
    task["id"]: task.get("max_steps", 20)
    for task in TASKS
}

TASK_DIFFICULTY = {
    task["id"]: task.get("difficulty", "medium")
    for task in TASKS
}

TASK_GRADERS = {
    task["id"]: task["grader_path"]
    for task in TASKS
}
