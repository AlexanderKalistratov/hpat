# *****************************************************************************
# Copyright (c) 2019, Intel Corporation All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#     Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

"""

| This file contains implementation of pandas general functions

"""

import pandas
import numpy
import numba

from textwrap import dedent

from collections import namedtuple

from numba import types, prange, typeof, numpy_support
from numba.typed import List, Dict
from numba.special import literally

from sdc.utilities.utils import sdc_overload, sdc_register_jitable
from sdc.utilities.sdc_typing_utils import is_default, TypeChecker
from sdc.hiframes.pd_dataframe_type import DataFrameType
from sdc.hiframes.pd_dataframe_ext import get_dataframe_data


def get_pool_size():
    return numba.config.NUMBA_NUM_THREADS


@sdc_overload(get_pool_size)
def get_pool_size_overload():
    pool_size = get_pool_size()
    def get_pool_size_impl():
        return pool_size

    return get_pool_size_impl


def get_chunks(size, pool_size=0):
    if pool_size == 0:
        pool_size = get_pool_size()

    chunk_size = size//pool_size + 1

    Chunk = namedtuple('start', 'stop')

    chunks = []

    for i in range(pool_size):
        start = min(i*chunk_size, size)
        stop = min((i + 1)*chunk_size, size)
        chunks.append(Chunk(start, stop))

    return chunks


@sdc_overload(get_chunks)
def get_chunks_overload(size, pool_size=0):
    Chunk = namedtuple('Chunk', ['start', 'stop'])

    def get_chunks_impl(size, pool_size=0):
        if pool_size == 0:
            pool_size = get_pool_size()

        chunk_size = size//pool_size + 1

        chunks = []

        for i in range(pool_size):
            start = min(i*chunk_size, size)
            stop = min((i + 1)*chunk_size, size)
            chunk = Chunk(start, stop)
            chunks.append(chunk)

        return chunks

    return get_chunks_impl


def quoted(var):
    return ["'" + val + "'" for val in var]


def extract_literal_value(var):
    if isinstance(var, (types.Tuple, types.UniTuple)):
        return [extract_literal_value(v) for v in var]
    elif isinstance(var, types.Literal):
        return var.literal_value
    else:
        raise ValueError(f'{var} has no literal values')

def make_function(func_name, func_text, global_vars):
    loc_vars = {}
    func_text = dedent(func_text)
    exec(func_text, global_vars, loc_vars)
    func = loc_vars[func_name]

    return func


def make_dataframe(cols, names, index=None):
    pass


def make_dataframe_codegen(col_names):
    data = ', '.join(f"'{name}': cols[{idx}]" for idx, name in enumerate(col_names))

    func_name = 'make_dataframe_impl'

    func_text = f"""
        def {func_name}(cols, names, index=None):
            # return pandas.DataFrame(data={{{data}}}, index=index)
            return pandas.DataFrame(data={{{data}}})
    """

    global_vars = {'pandas': pandas}

    return func_name, func_text, global_vars


@sdc_overload(make_dataframe)
def make_dataframe_overload(cols, names, index=None):
    names_values = extract_literal_value(names)
    # names_values = [name.literal_value for name in names]

    codegen_result = make_dataframe_codegen(names_values)

    return make_function(*codegen_result)


def extract_columns_as_tuple(dataframe, col_names=None):
    pass


def extract_columns_as_tuple_codegen(col_names):
    data = ', '.join(f"dataframe['{name}']._data" for name in col_names)

    func_name = 'extract_columns_as_tuple_impl'

    func_text = f"""
        def {func_name}(dataframe, col_names=None):
            return ({data}, )
    """

    return func_name, func_text, {}


@sdc_overload(extract_columns_as_tuple)
def extract_columns_as_tuple_overload(dataframe, col_names=None):
    if is_default(col_names):
        names_values = dataframe.columns
    else:
        names_values = [name.literal_value for name in col_names]

    codegen_result = extract_columns_as_tuple_codegen(names_values)

    return make_function(*codegen_result)


def get_line(columns, idx):
    pass


def get_line_codegen(columns):
    columns_count = len(columns)

    data = ', '.join(f'columns[{i}][idx]' for i in range(columns_count))

    func_name = 'get_line_impl'

    func_text = f"""
        def {func_name}(columns, idx):
            return ({data}, )
    """

    return func_name, func_text, {}


@sdc_overload(get_line, inline='always')
def get_line_overload(columns, idx):
    codegen_result = get_line_codegen(columns)

    return make_function(*codegen_result)


def set_line(lhs, lhs_idx, rhs, lhs_start = 0):
    pass


def set_line_codegen(rhs_size, lhs_start):
    func_name = 'set_line_impl'
    lhs_end = lhs_start + rhs_size

    lhs_vals = ', '.join([f'lhs[{lhs_pos}][lhs_idx]' for lhs_pos in range(lhs_start, lhs_end)])
    rhs_vals = ', '.join([f'rhs[{rhs_pos}]' for rhs_pos in range(rhs_size)])

    func_text = f"""
        def {func_name}(lhs, lhs_idx, rhs, lhs_start = 0):
            {lhs_vals} = {rhs_vals}
            return {lhs_vals}
    """

    return func_name, func_text, {}


@sdc_overload(set_line)
def set_line_overload(lhs, lhs_idx, rhs, lhs_start = 0):
    codegen_result = set_line_codegen(len(rhs), lhs_start.literal_value)

    return make_function(*codegen_result)


def unpack_tuple_vars(prefix, tpl):
    return [f'{prefix}_{i}' for i in range(len(tpl))]


def group_seq(keys):
    pass


@sdc_overload(group_seq)
def group_seq_overload(keys):
    import time
    def group_seq_impl(keys):
        first_key = get_line(keys, 0)
        result = {first_key: List.empty_list(types.int64)}

        first_column = keys[0]
        keys_len = len(first_column)

        get_line_time = 0
        get_time = 0
        set_time = 0
        for i in range(keys_len):
            start = time.time()
            key = get_line(keys, i)
            end = time.time()
            get_line_time += end - start
            items = result.get(key, List.empty_list(types.int64))
            start = time.time()
            get_time += start - end
            items.append(i)
            result[key] = items
            end = time.time()
            set_time += end - start

        print('get_line_time', get_line_time)
        print('get_time', get_time)
        print('set_time', set_time)

        return result

    return group_seq_impl


def inner_left_merge_keys(how, r_group, left_keys):
    pass


def inner_left_merge_keys_codegen(left_keys):
    func_name = 'inner_left_merge_keys_impl'

    keys_unpacked_list = unpack_tuple_vars('key', left_keys)
    keys_unpacked = ', '.join(keys_unpacked_list)

    func_text = f"""
        def {func_name}(how, r_group, left_keys):
            on_mismatch = 1 if how == 'left' else 0
            left_keys_len = len(left_keys[0])
            positions = numpy.empty(left_keys_len, dtype=numpy.int64)
            total_size = 0

            {keys_unpacked}, = left_keys
            for i in prange(left_keys_len):
                key = get_line(({keys_unpacked}, ), i)
                items = r_group.get(key, List.empty_list(types.int64))
                count = max(on_mismatch, len(items))
                total_size += count
                positions[i] = count

            positions = numpy.cumsum(positions)

            return total_size, positions
    """

    global_vars = {'get_line': get_line, 'List': List, 'types': types,
                   'prange': prange, 'numpy': numpy}

    return func_name, func_text, global_vars


@sdc_overload(inner_left_merge_keys)
def inner_left_merge_keys_overload(how, r_group, left_keys):
    codegen_result = inner_left_merge_keys_codegen(left_keys)

    return make_function(*codegen_result)


def make_tuple_with_values(size, value):
    pass


def make_tuple_with_values_codegen(size):
    func_name = 'make_tuple_with_values_impl'

    data = ', '.join(['value']*size.literal_value)

    func_text = f"""
        def {func_name}(size, value):
            return ({data}, )
    """

    return func_name, func_text, {}

@sdc_overload(make_tuple_with_values)
def make_tuple_with_values_overload(size, value):
    codegen_result = make_tuple_with_values_codegen(size)

    return make_function(*codegen_result)


def join_line(how, i, join_result, left_data, right_data, pos_start, count, items):
    pass


@sdc_overload(join_line)
def join_line_overload(how, i, join_result, left_data, right_data, pos_start, count, items):
    import time
    def join_line_impl(how, i, join_result, left_data, right_data, pos_start, count, items):
        left_data_size = len(left_data)
        right_data_size = len(right_data)
        if how == 'left':
            if count == 0:
                left_line = get_line(left_data, i)
                nans = make_tuple_with_values(right_data_size, numpy.nan)
                set_line(join_result, pos_start, left_line, 0)
                set_line(join_result, pos_start, nans, left_data_size)

        get_item_time = 0
        set_item_time = 0
        for j in range(count):
            start = time.time()
            left_line = get_line(left_data, i)
            right_line = get_line(right_data, items[j])
            end = time.time()
            get_item_time += end - start

            set_line(join_result, pos_start + j, left_line, 0)
            set_line(join_result, pos_start + j, right_line, left_data_size)
            start = time.time()
            set_item_time += start - end

        return get_item_time, set_item_time

    return join_line_impl


def columns_names_used_in_merge(left_columns, right_columns, left_keys, right_keys):
    lkey_set = set(left_keys)
    rkey_exclude = set([key for key in right_keys if key in lkey_set])

    left_names = left_columns
    right_names = [name for name in right_columns if name not in rkey_exclude]

    return left_names, right_names


def allocate_join_result(how, total_size, left, right, left_cols, right_cols):
    pass


def allocate_join_result_codegen(left_types, right_types):
    func_name = 'allocate_join_result_impl'

    left_data = [f'numpy.empty(total_size, dtype=numpy.{dtype})' for dtype in left_types]
    right_data = [f'numpy.empty(total_size, dtype=numpy.{dtype})' for dtype in right_types]

    all_data = ', '.join(left_data + right_data)

    func_text = f"""
        def {func_name}(how, total_size, left, right, left_cols, right_cols):
            return ({all_data}, )
    """

    return func_name, func_text, {'numpy': numpy}


@sdc_overload(allocate_join_result)
def allocate_join_result_overload(how, total_size, left, right, left_cols, right_cols):
    left_cols_vals = extract_literal_value(left_cols)
    right_cols_vals = extract_literal_value(right_cols)
    lcol_name_type = zip(left.data, left.columns)
    rcol_name_type = zip(right.data, right.columns)

    ldata = [data for data, name in lcol_name_type if name in left_cols_vals]
    rdata = [data for data, name in rcol_name_type if name in right_cols_vals]

    how_value = how.literal_value

    if how_value == 'outer':
        left_types = [types.float64 for d in ldata]
        right_types = [types.float64 for d in rdata]
    elif how_value == 'left':
        left_types = [d.dtype for d in ldata]
        right_types = [types.float64 for d in rdata]
    elif how_value == 'inner':
        left_types = [d.dtype for d in ldata]
        right_types = [d.dtype for d in rdata]

    codegen_result = allocate_join_result_codegen(left_types, right_types)

    return make_function(*codegen_result)


def left_cols_to_join(left, right, left_keys, right_keys):
    pass


def left_cols_to_join_codegen(cols):
    func_name = 'left_cols_to_join_impl'

    func_text = f"""
        def {func_name}(left, right, left_keys, right_keys):
            return ({cols}, )
    """

    return func_name, func_text, {}


@sdc_overload(left_cols_to_join)
def left_cols_to_join_overload(left, right, left_keys, right_keys):
    lkey_values = extract_literal_value(left_keys)
    rkey_values = extract_literal_value(right_keys)
    left_names, _ = columns_names_used_in_merge(left.columns, right.columns, lkey_values, rkey_values)

    left_names_str = ', '.join(quoted(left_names))

    codegen_result = left_cols_to_join_codegen(left_names_str)

    return make_function(*codegen_result)


def right_cols_to_join(left, right, left_keys, right_keys):
    pass


def right_cols_to_join_codegen(cols):
    func_name = 'right_cols_to_join_impl'

    func_text = f"""
        def {func_name}(left, right, left_keys, right_keys):
            return ({cols}, )
    """

    return func_name, func_text, {}


@sdc_overload(right_cols_to_join)
def right_cols_to_join_overload(left, right, left_keys, right_keys):
    lkey_values = extract_literal_value(left_keys)
    rkey_values = extract_literal_value(right_keys)
    _, right_names = columns_names_used_in_merge(left.columns, right.columns, lkey_values, rkey_values)

    right_names_str = ', '.join(quoted(right_names))

    codegen_result = right_cols_to_join_codegen(right_names_str)

    return make_function(*codegen_result)


def make_join_names(left_keys, right_keys, suffixes):
    pass


def make_join_names_codegen(all_names):
    func_name = 'make_join_names_impl'

    func_text = f"""
        def {func_name}(left_keys, right_keys, suffixes):
            return ({all_names}, )
    """

    return func_name, func_text, {}


@sdc_overload(make_join_names)
def make_join_names_overload(left_keys, right_keys, suffixes):
    lkey_values = extract_literal_value(left_keys)
    rkey_values = extract_literal_value(right_keys)

    left_names_set = set(lkey_values)
    right_names_set = set(rkey_values)

    def make_name(name, other_names, suffix):
        if name in other_names:
            return "'" + name + suffix + "'"

        return "'" + name + "'"

    left_names_list = [make_name(name, right_names_set, suffixes[0].literal_value) for name in lkey_values]
    right_names_list = [make_name(name, left_names_set, suffixes[1].literal_value) for name in rkey_values]

    all_names = ', '.join(left_names_list + right_names_list)

    codegen_result = make_join_names_codegen(all_names)

    return make_function(*codegen_result)


def inner_left_merge_data(how, left, right, r_group, left_keys, total_size, positions, join_result):
    pass


def inner_left_merge_data_codegen(how, left, right, r_group, left_keys, total_size, positions, join_result):
    func_name = 'inner_left_merge_data_impl'

    keys_unpacked_list = unpack_tuple_vars('key', left_keys)
    keys_unpacked = ', '.join(keys_unpacked_list)

    result_unpacked_list = unpack_tuple_vars('result', join_result)
    result_unpacked = ', '.join(result_unpacked_list)

    left_unpacked_list = unpack_tuple_vars('left_data', left)
    left_unpacked = ', '.join(left_unpacked_list)

    right_unpacked_list = unpack_tuple_vars('right_data', right)
    right_unpacked = ', '.join(right_unpacked_list)


    func_text = f"""
        def {func_name}(how, left, right, r_group, left_keys, total_size, positions, join_result):
            left_data = left
            right_data = right

            left_size = len(left_data[0])
            left_col_count = len(left_data)

            on_empty = 1 if how == 'left' else 0

            get_key_time = 0
            get_items_time = 0
            join_time = 0
            get_i_time = 0
            set_i_time = 0

            {keys_unpacked}, = left_keys
            {result_unpacked}, = join_result
            {left_unpacked}, = left_data
            {right_unpacked}, = right_data
            for i in range(left_size):
                start = time.time()
                key = get_line(({keys_unpacked}, ), i)
                end = time.time()
                get_key_time += end - start
                items = r_group.get(key, List.empty_list(types.int64))
                start = time.time()
                get_items_time += start - end
                count = len(items)
                pos_start = positions[i] - max(on_empty, count)

                t0, t1 = join_line(how, i, ({result_unpacked}, ), ({left_unpacked}, ),
                          ({right_unpacked}, ), pos_start, count, items)
                end = time.time()
                join_time += end - start
                get_i_time += t0
                set_i_time += t1

            print('get_key_time', get_key_time)
            print('get_items_time', get_items_time)
            print('join_time', join_time)
            print('get_i_time', get_i_time)
            print('set_i_time', set_i_time)

            return join_result
    """

    import time

    global_vars = {'make_dataframe': make_dataframe, 'make_join_names': make_join_names,
                   'allocate_join_result': allocate_join_result, 'extract_columns_as_tuple': extract_columns_as_tuple,
                   'join_line': join_line, 'prange': prange, 'get_line': get_line, 'List': List,
                   'types': types, 'time': time}

    return func_name, func_text, global_vars

@sdc_overload(inner_left_merge_data)
def inner_left_merge_data_overload(how, left, right, r_group, left_keys, total_size, positions, join_result):
    codegen_result = inner_left_merge_data_codegen(how, left, right, r_group, left_keys, total_size, positions, join_result)

    return make_function(*codegen_result)


def get_keys_names(side_on, on):
    pass


def get_keys_names_extract_keys(side_on, on):
    ty_checker = TypeChecker('Function merge().')

    on_present = not is_default(on)
    side_on_present = not is_default(side_on)

    if on_present and side_on_present:
        raise ValueError("""Either 'on' parameter or 'left_on'('left_index') and 'right_on'('right_index') must
            be specified at the same moment""")

    if not on_present and not side_on_present:
        raise ValueError("""Either 'on' parameter or 'left_on'('left_index') and 'right_on'('right_index') must
            be specified""")

    def extract_on(value):
        if isinstance(value, types.StringLiteral):
            return [value.literal_value]
        elif isinstance(value, types.Tuple):
            result = []
            for v in value:
                if isinstance(v, types.StringLiteral):
                    result.append(v.literal_value)
                else:
                    raise ValueError("Merge allowed only on literal values")

            return result
        else:
            raise ValueError("Unsupported type for '*on' parameter. Only literal/literal tuple is supported")

    if on_present:
        ty_checker.check(on, (types.StringLiteral, types.Tuple))

        return extract_on(on)

    ty_checker.check(side_on, (types.StringLiteral, types.Tuple))
    return extract_on(side_on)


def get_key_names_codegen(keys):
    keys_string = ', '.join(["'" + k + "'" for k in keys])

    func_name = 'get_key_names_impl'

    func_text = f"""
        def {func_name}(side_on, on):
            return ({keys_string}, )
    """

    return func_name, func_text, {}


@sdc_overload(get_keys_names)
def get_keys_names_overload(side_on, on):
    keys = get_keys_names_extract_keys(side_on, on)

    codegen_result = get_key_names_codegen(keys)

    return make_function(*codegen_result)


def extract_and_verify_merge_parameters(left, right, on=None, left_on=None,
                                        right_on=None, left_index=False, right_index=False,
                                        sort=False, suffixes=('_x', '_y'), copy=True,
                                        indicator=False, validate=None):
    ty_checker = TypeChecker('Function merge().')

    on_present = not is_default(on)
    left_on_present = not is_default(left_on)
    right_on_present = not is_default(right_on)
    left_index_present = not is_default(left_index, False)
    right_index_present = not is_default(right_index, False)

    if on_present and (left_on_present or left_index_present or right_on_present or right_index_present):
        raise ValueError("""Either 'on' parameter or 'left_on'('left_index') and 'right_on'('right_index') must
            be specified at the same moment""")

    left_keys = []
    right_keys = []

    if not left_index_present:
        left_keys.extend(get_keys_names_extract_keys(left_on, on))

    if not right_index_present:
        left_keys.extend(get_keys_names_extract_keys(right_on, on))

    return left_keys, right_keys


def dataframe_to_datatframe_merge_inner_left_overload(left, right, on=None, left_on=None,
                                                      right_on=None, left_index=False, right_index=False,
                                                      sort=False, suffixes=('_x', '_y'), copy=True,
                                                      indicator=False, validate=None):
    if is_default(suffixes, ('_x', '_y')):
        lsuffix = '_x'
        rsuffix = '_y'
    else:
        lsuffix = suffixes[0].literal_value
        rsuffix = suffixes[1].literal_value

    import time

    def df_df_merge_inner_left_impl(left, right, how='inner', on=None, left_on=None,
                                    right_on=None, left_index=False, right_index=False,
                                    sort=False, suffixes=('_x', '_y'), copy=True,
                                    indicator=False, validate=None):
        # literally(on)
        # literally(left_on)
        # literally(right_on)
        lkey_names = get_keys_names(left_on, on)
        rkey_names = get_keys_names(right_on, on)

        left_keys = extract_columns_as_tuple(left, lkey_names)
        right_keys = extract_columns_as_tuple(right, rkey_names)

        start = time.time()
        r_group = group_seq(right_keys)
        end = time.time()

        print('make rgroup', end - start)

        start = time.time()
        total_size, positions = inner_left_merge_keys(how, r_group, left_keys)
        end = time.time()

        print('join keys', end - start)
        left_cols_names = left_cols_to_join(left, right, lkey_names, rkey_names)
        right_cols_names = right_cols_to_join(left, right, lkey_names, rkey_names)
        join_names = make_join_names(left_cols_names, right_cols_names, (lsuffix, rsuffix))

        start = time.time()
        join_result = allocate_join_result(how, total_size, left, right, left_cols_names, right_cols_names)
        end = time.time()

        print('allocate result', end - start)


        left_data = extract_columns_as_tuple(left, left_cols_names)
        right_data = extract_columns_as_tuple(right, right_cols_names)

        start = time.time()
        join_result = inner_left_merge_data(how, left_data, right_data,
                                            r_group, left_keys, total_size,
                                            positions, join_result)
        end = time.time()

        print('merge data', end - start)

        return make_dataframe(join_result, join_names)

    return df_df_merge_inner_left_impl


def outer_group_keys(left_keys, right_keys):
    pass


@sdc_overload(outer_group_keys)
def outer_group_keys_overload(left_keys, right_keys):
    def outer_group_keys_impl(left_keys, right_keys):
        left_key = get_line(left_keys, 0)
        right_key = get_line(right_keys, 0)

        ldct = {left_key: List.empty_list(types.int64)}
        rdct = {right_key: List.empty_list(types.int64)}

        result = [ldct, rdct]
        keys = [left_keys, right_keys]

        for i in prange(2):
            result[i] = group_seq(keys[i])

        return result[0], result[1]

    return outer_group_keys_impl


def df_df_outer_join(left, right, left_on=None,
                     right_on=None, left_index=False, right_index=False,
                     sort=False, suffixes=('_x', '_y'), copy=True):
    pass


def outer_get_unique_keys(lgroup, rgroup):
    pass


@sdc_overload(outer_get_unique_keys)
def outer_get_unique_keys_overload(lgroup, rgroup):

    lkey_type = lgroup.key_type
    rkey_type = rgroup.key_type

    def outer_get_unique_keys_impl(lgroup, rgroup):
        lkeys = List.empty_list(lkey_type)
        rkeys = List.empty_list(rkey_type)

        groups = [lgroup, rgroup]
        keys = [lkeys, rkeys]

        for i in prange(2):
            gkeys = groups[i].keys()
            for key in gkeys:
                keys[i].append(key)
            # keys[i].append(list(groups[i].keys()))

        # lkeys = keys[0]
        # rkeys = keys[1]
        # lkeys = list(lgroup.keys())
        # rkeys = list(rgroup.keys())

        pool_size = get_pool_size()

        chunks = get_chunks(len(rgroup))
        chunks_count = len(chunks)

        extra_keys = [List.empty_list(rkey_type)]

        for i in range(1, chunks_count):
            extra_keys.append(List.empty_list(rkey_type))

        for i in prange(chunks_count):
            chunk = chunks[i]
            for j in range(chunk.start, chunk.stop):
                key = rkeys[j]
                key_in_lgroup = key in lgroup
                if key_in_lgroup == False:
                    extra_keys[i].append(key)

        extra_chunks = get_chunks(len(extra_keys), min(8, pool_size//2))

        extra_set = [set(extra_keys[0])]
        for i in range(1, len(extra_chunks)):
            chunk_ = extra_chunks[i]
            if chunk_.start < len(extra_keys):
                extra_set.append(set(extra_keys[chunk_.start]))
            else:
                extra_set.append(set(List.empty_list(rkey_type)))

        for i in range(len(extra_chunks)):
            est = extra_set[i]
            chunk__ = extra_chunks[i]
            for j in range(chunk__.start + 1, chunk__.stop):
                items = extra_keys[j]
                for item in items:
                    est.add(item)

        result_set = extra_set[0]

        for i in range(1, len(extra_chunks)):
            result_set.update(extra_set[i])

        # all_keys = lkeys + list(result_set)
        all_keys = lkeys

        for key in result_set:
            all_keys.append(key)

        # return all_keys
        return all_keys

    return outer_get_unique_keys_impl


@sdc_register_jitable
def outer_merge_keys_internal(lgroup, rgroup, all_keys):
    keys_size = len(all_keys)

    positions = numpy.empty(shape=keys_size, dtype=numpy.int64)

    for i in prange(keys_size):
        key = all_keys[i]
        lelems = lgroup.get(key, List.empty_list(types.int64))
        relems = rgroup.get(key, List.empty_list(types.int64))

        lelems_size = max(1, len(lelems))
        relems_size = max(1, len(relems))

        positions[i] = lelems_size*relems_size

    positions = numpy.cumsum(positions)

    total_size = positions[keys_size - 1]

    return all_keys, total_size, positions


def outer_merge_keys(lgroup, rgroup, sort):
    pass


@sdc_overload(outer_merge_keys)
def outer_merge_keys_overload(lgroup, rgroup, sort):

    if sort.literal_value:
        def outer_merge_keys_impl(lgroup, rgroup, sort):
            all_keys = outer_get_unique_keys(lgroup, rgroup)
            return outer_merge_keys_internal(lgroup, rgroup, sorted(all_keys))
    else:
        def outer_merge_keys_impl(lgroup, rgroup, sort):
            all_keys = outer_get_unique_keys(lgroup, rgroup)
            return outer_merge_keys_internal(lgroup, rgroup, all_keys)

    return outer_merge_keys_impl


def join_outer_for_key(result_start, left_data, right_data,
                       left_items, right_items, join_result):
    pass


@sdc_overload(join_outer_for_key)
def join_outer_for_key_overload(result_start, left_data, right_data,
                                left_items, right_items, join_result):
    def join_outer_for_key_impl(result_start, left_data, right_data,
                                left_items, right_items, join_result):
        left_data_size = len(left_data)
        right_data_size = len(right_data)

        if len(left_items) and len(right_items):
            for litem in left_items:
                for ritem in right_items:
                    left_line = get_line(left_data, litem)
                    right_line = get_line(right_data, ritem)
                    set_line(join_result, result_start, left_line, 0)
                    set_line(join_result, result_start, right_line, left_data_size)
        elif len(left_items):
            right_line = make_tuple_with_values(right_data_size, numpy.nan)
            for litem in left_items:
                left_line = get_line(left_data, litem)
                set_line(join_result, result_start, left_line, 0)
                set_line(join_result, result_start, right_line, left_data_size)
        else:
            left_line = make_tuple_with_values(left_data_size, numpy.nan)
            for ritem in right_items:
                right_line = get_line(right_data, ritem)
                set_line(join_result, result_start, left_line, 0)
                set_line(join_result, result_start, right_line, left_data_size)

    return join_outer_for_key_impl


def outer_merge_data(left_data, right_data,
                     lgroup, rgroup, all_keys,
                     positions, join_result):
    pass


def outer_merge_data_codegen(left, right, result):
    func_name = 'outer_merge_data__impl'

    result_unpacked_list = unpack_tuple_vars('result', result)
    result_unpacked = ', '.join(result_unpacked_list)

    left_unpacked_list = unpack_tuple_vars('left_data', left)
    left_unpacked = ', '.join(left_unpacked_list)

    right_unpacked_list = unpack_tuple_vars('right_data', right)
    right_unpacked = ', '.join(right_unpacked_list)

    func_text = f"""
        def {func_name}(left_data, right_data, lgroup, rgroup, all_keys, positions, join_result):
            all_keys_size = len(all_keys)

            {result_unpacked}, = join_result
            {left_unpacked}, = left_data
            {right_unpacked}, = right_data
            for i in prange(all_keys_size):
                key = all_keys[i]
                pos_end = positions[i]
                left_items = lgroup.get(key, List.empty_list(types.int64))
                right_items = rgroup.get(key, List.empty_list(types.int64))
                key_items_count = max(1, len(left_items)) * max(1, len(right_items))
                result_start = pos_end - key_items_count

                join_outer_for_key(result_start, ({left_unpacked}, ), ({right_unpacked}, ),
                                   left_items, right_items, ({result_unpacked}, ))

            return join_result
    """

    global_vars = {'join_outer_for_key': join_outer_for_key, 'prange': prange,
                   'List': List, 'types': types}

    return func_name, func_text, global_vars


@sdc_overload(outer_merge_data)
def outer_merge_data_overload(left_data, right_data,
                              lgroup, rgroup, all_keys,
                              positions, join_result):

    codegen_result = outer_merge_data_codegen(left_data, right_data, join_result)
    # def outer_merge_data_impl(left_data, right_data,
    #                           lgroup, rgroup, all_keys,
    #                           positions, join_result):
    #     all_keys_size = len(all_keys)

    #     for i in prange(all_keys_size):
    #         key = all_keys[i]
    #         pos_end = positions[i]
    #         left_items = lgroup.get(key, List.empty_list(types.int64))
    #         right_items = rgroup.get(key, List.empty_list(types.int64))
    #         key_items_count = max(1, len(left_items)) * max(1, len(right_items))
    #         result_start = pos_end - key_items_count

    #         join_outer_for_key(result_start, left_data, right_data,
    #                            left_items, right_items, join_result)

    # return outer_merge_data_impl

    return make_function(*codegen_result)


@sdc_overload(df_df_outer_join)
def df_df_outer_join_overload(left, right, left_on=None,
                              right_on=None, left_index=False, right_index=False,
                              sort=False, suffixes=('_x', '_y'), copy=True):

    if is_default(suffixes, ('_x', '_y')):
        lsuffix = '_x'
        rsuffix = '_y'
    else:
        lsuffix = suffixes[0].literal_value
        rsuffix = suffixes[1].literal_value

    if is_default(sort, False):
        sort_value = False
    else:
        sort_value = sort.literal_value

    def df_df_outer_join_impl(left, right, left_on=None,
                              right_on=None, left_index=False, right_index=False,
                              sort=False, suffixes=('_x', '_y'), copy=True):
        lkey_names = left_on
        rkey_names = right_on

        left_keys = extract_columns_as_tuple(left, lkey_names)
        right_keys = extract_columns_as_tuple(right, rkey_names)

        lgroup, rgroup = outer_group_keys(left_keys, right_keys)

        all_keys, total_size, positions = outer_merge_keys(lgroup, rgroup, sort_value)

        left_cols_names = left_cols_to_join(left, right, lkey_names, rkey_names)
        right_cols_names = right_cols_to_join(left, right, lkey_names, rkey_names)
        join_names = make_join_names(left_cols_names, right_cols_names, (lsuffix, rsuffix))

        join_result = allocate_join_result('outer', total_size, left, right, left_cols_names, right_cols_names)

        left_data = extract_columns_as_tuple(left, left_cols_names)
        right_data = extract_columns_as_tuple(right, right_cols_names)

        join_result = outer_merge_data(left_data, right_data,
                                       lgroup, rgroup, all_keys,
                                       positions, join_result)

        return make_dataframe(join_result, join_names)

    return df_df_outer_join_impl



def dataframe_to_datatframe_merge_outer_overload(left, right, on=None, left_on=None,
                                                 right_on=None, left_index=False, right_index=False,
                                                 sort=False, suffixes=('_x', '_y'), copy=True,
                                                 indicator=False, validate=None):

    if is_default(suffixes, ('_x', '_y')):
        lsuffix = '_x'
        rsuffix = '_y'
    else:
        lsuffix = suffixes[0].literal_value
        rsuffix = suffixes[1].literal_value

    def df_df_merge_outer_join_impl(left, right, how='inner', on=None, left_on=None,
                                    right_on=None, left_index=False, right_index=False,
                                    sort=False, suffixes=('_x', '_y'), copy=True,
                                    indicator=False, validate=None):
        lkey_names = get_keys_names(left_on, on)
        rkey_names = get_keys_names(right_on, on)

        return df_df_outer_join(left, right, lkey_names, rkey_names,
                                left_index, right_index, True,
                                (lsuffix, rsuffix), copy)


    return df_df_merge_outer_join_impl


def dataframe_to_dataframe_merge(left, right, how='inner', on=None, left_on=None,
                                 right_on=None, left_index=False, right_index=False,
                                 sort=False, suffixes=('_x', '_y'), copy=True,
                                 indicator=False, validate=None):

    ty_checker = TypeChecker('Function merge().')

    if is_default(how, 'inner'):
        how_value = 'inner'
    else:
        ty_checker.check(how, types.StringLiteral)
        how_value = how.literal_value

    if how_value == 'inner':
        return dataframe_to_datatframe_merge_inner_left_overload(left, right, on,
                                                                 left_on, right_on,
                                                                 left_index, right_index,
                                                                 sort, suffixes, copy,
                                                                 indicator, validate)
    elif how_value == 'left':
        return dataframe_to_datatframe_merge_inner_left_overload(left, right, on,
                                                                 left_on, right_on,
                                                                 left_index, right_index,
                                                                 sort, suffixes, copy,
                                                                 indicator, validate)

    elif how_value == 'outer':
        return dataframe_to_datatframe_merge_outer_overload(right, left, on,
                                                            left_on, right_on,
                                                            left_index, right_index,
                                                            sort, suffixes, copy,
                                                            indicator, validate)
    else:
        raise ValueError(f"Unsupported 'how' parameter value. "
                         f"Expected: 'inner', 'left', or 'outer'. Got {how_value}")

    return dataframe_to_dataframe_merge_impl


@sdc_overload(pandas.merge)
def sdc_pandas_general_merge(left, right, how='inner', on=None, left_on=None,
                             right_on=None, left_index=False, right_index=False,
                             sort=False, suffixes=('_x', '_y'), copy=True,
                             indicator=False, validate=None):

    if isinstance(left, DataFrameType) and isinstance(right, DataFrameType):
        return dataframe_to_dataframe_merge(left, right, how, on, left_on,
                                            right_on, left_index, right_index,
                                            sort, suffixes, copy,
                                            indicator, validate)

    return None
