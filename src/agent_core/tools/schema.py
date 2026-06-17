from __future__ import annotations

from typing import Any


def validate_schema_subset(schema: Any, *, path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise ValueError(f"tool schema at {path} must be an object")
    unsupported = {"$ref", "oneOf", "anyOf", "allOf", "if", "then", "else", "patternProperties"}
    for key in unsupported:
        if key in schema:
            raise ValueError(f"unsupported tool schema keyword at {path}: {key}")
    schema_type = schema.get("type")
    allowed_types = {"object", "string", "number", "integer", "boolean", "array", "null"}
    if isinstance(schema_type, list):
        invalid = [item for item in schema_type if item not in allowed_types]
        if invalid:
            raise ValueError(f"unsupported schema type at {path}: {invalid[0]}")
    elif schema_type is not None and schema_type not in allowed_types:
        raise ValueError(f"unsupported schema type at {path}: {schema_type}")
    properties = schema.get("properties") or {}
    if properties and not isinstance(properties, dict):
        raise ValueError(f"schema properties at {path} must be an object")
    for name, child in properties.items():
        validate_schema_subset(child, path=f"{path}.properties.{name}")
    items = schema.get("items")
    if isinstance(items, dict):
        validate_schema_subset(items, path=f"{path}.items")
    elif isinstance(items, list):
        raise ValueError(f"tuple-style array schemas are not supported at {path}.items")


def validate_arguments(arguments: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    schema_type = schema.get("type")
    allowed_types = schema_type if isinstance(schema_type, list) else ([schema_type] if schema_type is not None else [])
    if "null" in allowed_types and arguments is None:
        return
    non_null_types = [item for item in allowed_types if item != "null"]
    if non_null_types and not any(matches_json_type(arguments, item) for item in non_null_types):
        raise ValueError(f"{path} must be {type_label(non_null_types)}")
    if schema.get("enum") is not None and arguments not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    effective_type = non_null_types[0] if non_null_types else None
    if effective_type == "object" or (effective_type is None and isinstance(arguments, dict)):
        if not isinstance(arguments, dict):
            raise ValueError(f"{path} must be object")
        required = schema.get("required") or []
        for key in required:
            if key not in arguments:
                raise ValueError(f"{path}.{key} is required")
        properties = schema.get("properties") or {}
        if properties and schema.get("additionalProperties", False) is not True:
            unknown = sorted(key for key in arguments if key not in properties)
            if unknown:
                raise ValueError(f"{path} contains unknown field: {unknown[0]}")
        for key, value in arguments.items():
            if key in properties:
                validate_arguments(value, properties[key], path=f"{path}.{key}")
    elif effective_type == "array" or (effective_type is None and isinstance(arguments, list)):
        if not isinstance(arguments, list):
            raise ValueError(f"{path} must be array")
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, value in enumerate(arguments):
                validate_arguments(value, items_schema, path=f"{path}[{index}]")


def matches_json_type(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return True


def type_label(types: list[str]) -> str:
    return " or ".join(types)
