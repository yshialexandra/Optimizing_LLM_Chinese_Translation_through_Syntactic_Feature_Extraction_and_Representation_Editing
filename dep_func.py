def depTripleFuncLex(sentence, head, relation, dependent):    # VV NSUBJ 我
    head_list = []
    # print(sentence)
    for item in sentence:
        if item.xpos == head or item.text == head and item.xpos == 'PN':
            head_list.append(item.id)

    for index in head_list:
        for item in sentence:
            if item.head == index and item.text == dependent and item.deprel == relation:
                return True
    
    return False
