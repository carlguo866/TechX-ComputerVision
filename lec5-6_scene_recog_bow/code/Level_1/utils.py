import matplotlib.pyplot as plt

def print_data_stats(data,title='Data statistics'):
    total = 0
    for k,v in data.items():
        total += len(v)
    print('{}: loaded {} classes, a total of {} images.'.format(title, len(data), total))

def humanize_time(secs):
    """
    Extracted from http://testingreflections.com/node/6534
    """
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02f' % (hours, mins, secs)

def show_conf_mat(confusion_matrix, id2name=None):
    """
    Show a windows with a color image for a confusion matrix
    Args:
        confusion_matrix (NumPy Array): The matrix to be shown.
    Returns:
        void
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix, interpolation='nearest')
    fig.colorbar(cax)

    if not id2name is None:
        class_names = [id2name[i] for i in range(confusion_matrix.shape[0])]
        ax.set_xticklabels(['']+class_names)
        ax.set_yticklabels(['']+class_names)

    plt.show()