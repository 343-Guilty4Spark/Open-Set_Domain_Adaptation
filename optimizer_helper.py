from torch import optim

def get_optim_and_scheduler(feature_extractor, obj_cls, self_cls, epochs, mil_sched, lr, weight_decay, train_all):

    params = list(obj_cls.parameters())
    for i in self_cls:
        params += list(i.parameters())
    if train_all:
        params += list(feature_extractor.parameters())

    optimizer = optim.SGD(params, weight_decay=weight_decay, momentum=.9, lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mil_sched, gamma=0.1, verbose=True)

    return optimizer, scheduler