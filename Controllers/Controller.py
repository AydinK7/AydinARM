class Controller:
    def start() -> None: ...
    def stop() -> None: ...   
    
    def is_label_in_universe(self, label: str) -> bool: ...
    """Returns True if the label is something this controler can see, else false."""
    
    def set_target_label(self, label: str) -> bool: ...
    """This controller will only target objects with the specified label."""
    
    def get_target_label(self) -> str: ...
    """Returns the label of the object that the controller is currently targeting."""
    
    def get_visible_object_labels(self) -> list[str]: ...
    """Returns a list of identifiers of objects that are visible to the arm"""
    
    def get_visible_object_labels_detailed(self) -> list[str]: ...
    """Returns a list of objects that are visible to the arm, including metadata"""
    
    def get_all_posible_labels(self) -> list[str]: ...
    """Returns a list of all possible labels that this controller can see, even if they are not currently visible."""
    