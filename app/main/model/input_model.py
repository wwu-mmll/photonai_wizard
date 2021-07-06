class BasicInput:

    def __init__(self, title: str, question: str, fieldset_id: str, input_type: str,
                 property_name: str, find_object: bool = False,
                 is_required: bool = True, default_value=None, full_width:bool =False):
        self.title = title
        self.question = question
        self.fieldset_id = fieldset_id
        self.input_type = input_type
        self.property_name = property_name
        self.find_object = find_object
        self.is_required = is_required
        self.default_value = default_value
        self.full_width = full_width


class SelectionBox(BasicInput):

    def __init__(self, title: str, question: str, items: list, radio_buttons: bool, fieldset_id: str,
                 property_name: str, object_type: type, has_parameters: bool = False,
                 find_many: bool = False, has_test_disabled=False, is_required: bool = True,
                 constraint_list: list = [], default_value=None,
                 hyperparameter_dict: dict=None,
                 is_active: bool = False, position: int = 0, test_disabled: bool = False):

        super().__init__(title, question, fieldset_id, "selection_box", property_name, True,
                         is_required=is_required, default_value=default_value)
        self.type_selection_box = True
        self.items = items
        self.radio_buttons = radio_buttons
        self.object_type = object_type
        self.has_parameters = has_parameters
        self.find_many = find_many
        self.has_test_disabled = has_test_disabled
        self.constraint_list = constraint_list
        self.hyperparameter_dict = hyperparameter_dict
        self.is_active = is_active
        self.position = position
        self.test_disabled = test_disabled


class InputBox(BasicInput):
    def __init__(self, title: str, question: str, fieldset_id: str, property_name: str,
                 is_required: bool = True, default_value=None, possible_values: list = [],
                 multi_select=True, full_width: bool = False):
        super().__init__(title, question, fieldset_id, "textfield",
                         property_name=property_name, is_required=is_required,
                         default_value=default_value, full_width = full_width)
        self.possible_values = possible_values
        self.multi_select = multi_select



