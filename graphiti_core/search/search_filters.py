"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.neug.dialect import esc, iso, neug_in_filter
from graphiti_core.helpers import validate_node_labels


class ComparisonOperator(Enum):
    equals = '='
    not_equals = '<>'
    greater_than = '>'
    less_than = '<'
    greater_than_equal = '>='
    less_than_equal = '<='
    is_null = 'IS NULL'
    is_not_null = 'IS NOT NULL'


class DateFilter(BaseModel):
    date: datetime | None = Field(default=None, description='A datetime to filter on')
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for date filter'
    )


class PropertyFilter(BaseModel):
    property_name: str = Field(description='Property name')
    property_value: str | int | float | None = Field(
        default=None, description='Value you want to match on for the property'
    )
    comparison_operator: ComparisonOperator = Field(
        description='Comparison operator for the property'
    )


class SearchFilters(BaseModel):
    node_labels: list[str] | None = Field(
        default=None, description='List of node labels to filter on'
    )
    edge_types: list[str] | None = Field(
        default=None, description='List of edge types to filter on'
    )
    valid_at: list[list[DateFilter]] | None = Field(default=None)
    invalid_at: list[list[DateFilter]] | None = Field(default=None)
    created_at: list[list[DateFilter]] | None = Field(default=None)
    expired_at: list[list[DateFilter]] | None = Field(default=None)
    edge_uuids: list[str] | None = Field(default=None)
    property_filters: list[PropertyFilter] | None = Field(default=None)

    @field_validator('node_labels')
    @classmethod
    def validate_node_label_filters(cls, value: list[str] | None) -> list[str] | None:
        validate_node_labels(value)
        return value


def cypher_to_opensearch_operator(op: ComparisonOperator) -> str:
    mapping = {
        ComparisonOperator.greater_than: 'gt',
        ComparisonOperator.less_than: 'lt',
        ComparisonOperator.greater_than_equal: 'gte',
        ComparisonOperator.less_than_equal: 'lte',
    }
    return mapping.get(op, op.value)


def node_search_filter_query_constructor(
    filters: SearchFilters,
    provider: GraphProvider,
) -> tuple[list[str], dict[str, Any]]:
    filter_queries: list[str] = []
    filter_params: dict[str, Any] = {}

    if filters.node_labels is not None:
        # Defense-in-depth for model_construct()/other validation bypasses.
        validate_node_labels(filters.node_labels)
        if provider == GraphProvider.KUZU:
            node_label_filter = 'list_has_all(n.labels, $labels)'
            filter_params['labels'] = filters.node_labels
        elif provider == GraphProvider.NEUG:
            # labels is a STRING[] column; bound list params are unusable, so
            # chain one list_contains() call per label with inlined literals.
            node_label_filter = ' AND '.join(
                f'list_contains(n.labels, {esc(label)})' for label in filters.node_labels
            )
        else:
            node_labels = '|'.join(filters.node_labels)
            node_label_filter = 'n:' + node_labels
        filter_queries.append(node_label_filter)

    return filter_queries, filter_params


def date_filter_query_constructor(
    value_name: str, param_name: str, operator: ComparisonOperator
) -> str:
    query = '(' + value_name + ' '

    if operator == ComparisonOperator.is_null or operator == ComparisonOperator.is_not_null:
        query += operator.value + ')'
    else:
        query += operator.value + ' ' + param_name + ')'

    return query


def date_filter_query_constructor_neug(
    value_name: str, param_name: str, operator: ComparisonOperator
) -> str:
    # NeuG stores unset STRING columns as '' rather than NULL, so null checks
    # become empty-string comparisons.
    query = '(' + value_name + ' '

    if operator == ComparisonOperator.is_null:
        query += "= '')"
    elif operator == ComparisonOperator.is_not_null:
        query += "<> '')"
    else:
        query += operator.value + ' ' + param_name + ')'

    return query


def neug_node_label_filter(labels: list[str]) -> str:
    """AND-chain list_contains() over both endpoints' labels (NEUG).

    ``labels`` is a STRING[] column and bound list params are unusable for it,
    so each label is inlined as a literal. The query must bind ``n`` and ``m``
    to the source/target Entity nodes (only reachable via a join, since the
    EdgeDoc mirror carries no label columns).
    """
    return ' AND '.join(
        f'list_contains(n.labels, {esc(label)}) AND list_contains(m.labels, {esc(label)})'
        for label in labels
    )


def edge_search_filter_query_constructor(
    filters: SearchFilters,
    provider: GraphProvider,
    edge_alias: str = 'e',
    emit_node_labels: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Edge filters. ``edge_alias`` retargets every edge-column predicate (the
    NEUG EdgeDoc-mirror path passes ``d``); ``emit_node_labels=False`` drops the
    node_labels predicate, which needs n/m and so cannot ride an EdgeDoc-only
    prefilter (the caller joins for it separately).
    """
    filter_queries: list[str] = []
    filter_params: dict[str, Any] = {}

    if provider == GraphProvider.NEUG:
        date_constructor = date_filter_query_constructor_neug
    else:
        date_constructor = date_filter_query_constructor

    def date_param_value(date_filter: DateFilter) -> Any:
        # NeuG stores datetimes as fixed-width ISO-8601 strings, so filter
        # values must be serialized to match.
        if provider == GraphProvider.NEUG:
            return iso(date_filter.date)
        return date_filter.date

    if filters.edge_types is not None:
        if provider == GraphProvider.NEUG:
            filter_queries.append(
                neug_in_filter(f'{edge_alias}.name', filters.edge_types, filter_params)
            )
        else:
            edge_types = filters.edge_types
            filter_queries.append(f'{edge_alias}.name in $edge_types')
            filter_params['edge_types'] = edge_types

    if filters.edge_uuids is not None:
        if provider == GraphProvider.NEUG:
            filter_queries.append(
                neug_in_filter(f'{edge_alias}.uuid', filters.edge_uuids, filter_params)
            )
        else:
            filter_queries.append(f'{edge_alias}.uuid in $edge_uuids')
            filter_params['edge_uuids'] = filters.edge_uuids

    if filters.node_labels is not None and emit_node_labels:
        # Defense-in-depth for model_construct()/other validation bypasses.
        validate_node_labels(filters.node_labels)
        if provider == GraphProvider.KUZU:
            node_label_filter = (
                'list_has_all(n.labels, $labels) AND list_has_all(m.labels, $labels)'
            )
            filter_params['labels'] = filters.node_labels
        elif provider == GraphProvider.NEUG:
            node_label_filter = neug_node_label_filter(filters.node_labels)
        else:
            node_labels = '|'.join(filters.node_labels)
            node_label_filter = 'n:' + node_labels + ' AND m:' + node_labels
        filter_queries.append(node_label_filter)

    if filters.valid_at is not None:
        valid_at_filter = '('
        for i, or_list in enumerate(filters.valid_at):
            for j, date_filter in enumerate(or_list):
                if date_filter.comparison_operator not in [
                    ComparisonOperator.is_null,
                    ComparisonOperator.is_not_null,
                ]:
                    filter_params['valid_at_' + str(j)] = date_param_value(date_filter)

            and_filters = [
                date_constructor(
                    f'{edge_alias}.valid_at', f'$valid_at_{j}', date_filter.comparison_operator
                )
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            valid_at_filter += and_filter_query

            if i == len(filters.valid_at) - 1:
                valid_at_filter += ')'
            else:
                valid_at_filter += ' OR '

        filter_queries.append(valid_at_filter)

    if filters.invalid_at is not None:
        invalid_at_filter = '('
        for i, or_list in enumerate(filters.invalid_at):
            for j, date_filter in enumerate(or_list):
                if date_filter.comparison_operator not in [
                    ComparisonOperator.is_null,
                    ComparisonOperator.is_not_null,
                ]:
                    filter_params['invalid_at_' + str(j)] = date_param_value(date_filter)

            and_filters = [
                date_constructor(
                    f'{edge_alias}.invalid_at', f'$invalid_at_{j}', date_filter.comparison_operator
                )
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            invalid_at_filter += and_filter_query

            if i == len(filters.invalid_at) - 1:
                invalid_at_filter += ')'
            else:
                invalid_at_filter += ' OR '

        filter_queries.append(invalid_at_filter)

    if filters.created_at is not None:
        created_at_filter = '('
        for i, or_list in enumerate(filters.created_at):
            for j, date_filter in enumerate(or_list):
                if date_filter.comparison_operator not in [
                    ComparisonOperator.is_null,
                    ComparisonOperator.is_not_null,
                ]:
                    filter_params['created_at_' + str(j)] = date_param_value(date_filter)

            and_filters = [
                date_constructor(
                    f'{edge_alias}.created_at', f'$created_at_{j}', date_filter.comparison_operator
                )
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            created_at_filter += and_filter_query

            if i == len(filters.created_at) - 1:
                created_at_filter += ')'
            else:
                created_at_filter += ' OR '

        filter_queries.append(created_at_filter)

    if filters.expired_at is not None:
        expired_at_filter = '('
        for i, or_list in enumerate(filters.expired_at):
            for j, date_filter in enumerate(or_list):
                if date_filter.comparison_operator not in [
                    ComparisonOperator.is_null,
                    ComparisonOperator.is_not_null,
                ]:
                    filter_params['expired_at_' + str(j)] = date_param_value(date_filter)

            and_filters = [
                date_constructor(
                    f'{edge_alias}.expired_at', f'$expired_at_{j}', date_filter.comparison_operator
                )
                for j, date_filter in enumerate(or_list)
            ]
            and_filter_query = ''
            for j, and_filter in enumerate(and_filters):
                and_filter_query += and_filter
                if j != len(and_filters) - 1:
                    and_filter_query += ' AND '

            expired_at_filter += and_filter_query

            if i == len(filters.expired_at) - 1:
                expired_at_filter += ')'
            else:
                expired_at_filter += ' OR '

        filter_queries.append(expired_at_filter)

    return filter_queries, filter_params
