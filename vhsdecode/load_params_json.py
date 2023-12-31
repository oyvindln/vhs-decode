import json


def _override_group(json_input, group, group_name, logger):
    """Go through a group of parameters and replace the ones found in 'json_input'"""
    changed = {}
    if group_name in json_input:
        for item in json_input[group_name]:
            if item in group:
                changed[item] = json_input[group_name][item]
                group[item] = json_input[group_name][item]
    if len(changed) > 0:
        logger.debug("Changed %s in %s", changed, group_name)


def override_params(sys_params, rf_params, file, logger):
    """Override parameters with ones from the specified json file."""
    json_input = json.load(file)
    _override_group(json_input, sys_params, "sys_params", logger)
    _override_group(json_input, rf_params, "rf_params", logger)
