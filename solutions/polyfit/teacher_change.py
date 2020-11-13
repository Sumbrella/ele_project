change_weights = [
    100,
    10,
    100,
    10,
    100,
    10,
    100,
    10,
    10
]


def change_teacher(teachers):
    for id, teacher in enumerate(teachers):
        for index, item in enumerate(teacher):
            teachers[id][index] = change_weights[index] * item

    return teachers


def back_change(labels):
    for id, label in enumerate(labels):
        for index, item in enumerate(label):
            labels[id][index] = item / change_weights[index]

    return labels