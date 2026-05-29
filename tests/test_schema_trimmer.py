"""Tests for the OpenAPI / GraphQL schema trimmer."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tokenshrink.compressors.schema_trimmer import (
    trim_openapi,
    trim_graphql,
    trim_schema,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

OPENAPI_SCHEMA = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0"},
    "paths": {
        "/users": {
            "get": {
                "summary": "List users",
                "description": "Returns all registered users in the system.",
                "tags": ["users"],
                "responses": {
                    "200": {"description": "OK"},
                    "422": {"description": "Validation error"},
                },
            },
            "post": {
                "summary": "Create user",
                "description": "Create a new user account.",
                "tags": ["users"],
                "responses": {"201": {"description": "Created"}},
            },
        },
        "/products": {
            "get": {
                "summary": "List products",
                "description": "Returns all available products.",
                "tags": ["products"],
                "responses": {"200": {"description": "OK"}},
            },
        },
        "/orders": {
            "post": {
                "summary": "Place order",
                "description": "Place a new product order.",
                "tags": ["orders"],
                "responses": {"201": {"description": "Created"}},
                "deprecated": True,
            },
        },
    },
}

GRAPHQL_SDL = '''\
"""
This is a very long docstring that exceeds the eighty character limit for the User type in GraphQL.
"""
type User {
  id: ID!
  name: String!
  """Short doc"""
  email: String
  oldField: String @deprecated(reason: "use email instead")
}
'''


# ── OpenAPI tests ─────────────────────────────────────────────────────────────

class TestTrimOpenAPI:
    def test_keeps_relevant_operations(self):
        result = trim_openapi(OPENAPI_SCHEMA, task="list all users", top_k=2)
        paths = result["paths"]
        # /users GET should score highest for "list all users"
        assert "/users" in paths
        assert "get" in paths["/users"]

    def test_deprecated_excluded(self):
        result = trim_openapi(OPENAPI_SCHEMA, task="place order", top_k=5)
        # /orders POST is deprecated — should be excluded
        paths = result["paths"]
        if "/orders" in paths:
            assert "post" not in paths["/orders"]

    def test_422_stripped(self):
        result = trim_openapi(OPENAPI_SCHEMA, task="list users", top_k=5)
        for path_item in result.get("paths", {}).values():
            for op in path_item.values():
                assert "422" not in op.get("responses", {})

    def test_top_k_limits_operations(self):
        result = trim_openapi(OPENAPI_SCHEMA, task="anything", top_k=1)
        total_ops = sum(len(v) for v in result["paths"].values())
        assert total_ops <= 1

    def test_info_preserved(self):
        result = trim_openapi(OPENAPI_SCHEMA, task="users", top_k=5)
        assert result["info"]["title"] == "Test API"

    def test_long_description_truncated(self):
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/things": {
                    "get": {
                        "summary": "List things",
                        "description": "A" * 200,
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        result = trim_openapi(schema, task="list things", top_k=1)
        desc = result["paths"]["/things"]["get"].get("description", "")
        assert len(desc) <= 85  # 80 + ellipsis

    def test_example_fields_stripped(self):
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/items": {
                    "get": {
                        "summary": "Get items",
                        "example": {"id": 1},
                        "examples": {"foo": "bar"},
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        result = trim_openapi(schema, task="get items", top_k=1)
        op = result["paths"]["/items"]["get"]
        assert "example" not in op
        assert "examples" not in op

    def test_extension_keys_stripped(self):
        schema = {
            "openapi": "3.0.0",
            "paths": {
                "/items": {
                    "get": {
                        "summary": "Get items",
                        "x-custom-header": "secret",
                        "responses": {"200": {"description": "OK"}},
                    }
                }
            },
        }
        result = trim_openapi(schema, task="get items", top_k=1)
        op = result["paths"]["/items"]["get"]
        assert "x-custom-header" not in op


# ── GraphQL tests ─────────────────────────────────────────────────────────────

class TestTrimGraphQL:
    def test_long_docstring_truncated(self):
        result = trim_graphql(GRAPHQL_SDL, task="user")
        # The long User docstring should be shortened
        assert "eighty character" not in result or '"""' in result
        assert len(result) < len(GRAPHQL_SDL)

    def test_short_docstring_preserved(self):
        result = trim_graphql(GRAPHQL_SDL, task="user")
        assert "Short doc" in result

    def test_deprecated_fields_removed(self):
        result = trim_graphql(GRAPHQL_SDL, task="user")
        assert "@deprecated" not in result


# ── Auto-detect tests ─────────────────────────────────────────────────────────

class TestTrimSchema:
    def test_json_openapi_detected(self):
        result = trim_schema(json.dumps(OPENAPI_SCHEMA), task="list users", top_k=2)
        data = json.loads(result)
        assert "paths" in data

    def test_graphql_detected(self):
        result = trim_schema(GRAPHQL_SDL, task="user fields")
        # Should return stripped SDL text (not JSON)
        assert "type User" in result

    def test_unknown_format_passthrough(self):
        plain = "just some random text"
        result = trim_schema(plain, task="anything")
        assert result == plain
