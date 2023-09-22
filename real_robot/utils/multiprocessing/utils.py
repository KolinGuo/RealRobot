from .shared_object import SharedObject


class SharedObjectDefaultDict(dict):
    """This defaultdict helps to store SharedObject by name (only known at runtime)
    so we don't need to frequently create SharedObject
    """

    def __missing__(self, so_name: str) -> SharedObject:
        so = self[so_name] = SharedObject(so_name)
        return so
