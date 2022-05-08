import numpy as np


def _format_bytearray(data: bytearray, display_limit: int = 16) -> str:
    if data:
        bytes_repr = f"[ {bytes(data[:display_limit]).hex(' ').upper()}"
        if len(data) > display_limit:
            bytes_repr += f'  ({len(data) - display_limit} more bytes...)'
        bytes_repr += ' ]'
    else:
        bytes_repr = '[ empty ]'

    return bytes_repr


def _format_numpy(data: np.array, display_limit: int = 16) -> str:
    if len(data) > 0:
        arr_repr = f'{data[:display_limit]}'
        if len(data) > display_limit:
            arr_repr += f'   ({len(data) - display_limit} more items...)'

        if '\n' in arr_repr:
            arr_repr = f'\n{arr_repr}'
    else:
        arr_repr = '[ empty ]'

    return arr_repr
