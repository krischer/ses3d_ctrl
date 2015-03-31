import os

TEMPLATE_FOLDER = os.path.join(os.path.dirname(__file__), "templates")


def get_template(name):
    """
    Returns the template filename.
    """
    filename = os.path.join(TEMPLATE_FOLDER,
                            name + os.path.extsep + "template")
    if not os.path.exists(filename):
        raise ValueError("Cannot find template '%s'." % name)

    return filename
