# Keywords


| Name                             | Meaning                                                      |
| -------------------------------- | ------------------------------------------------------------ |
| **unlabel**, **ignore** [1]      | Classes whose names contain the reserved keywords are ignored during any category-wise operation. |
| **background** [1]               | Background class                                             |
| **label**, **seg**, **mask** [2] | Channels identified this way are treated as labels: e.g., resampling with nearest-neighbor interpolation. |

**Note** 
The keywords **above** are **case-insensitive**; a channel is flagged as soon as the keyword appears **anywhere** in its name.



| Name       | Meaning                                       |
| ---------- | --------------------------------------------- |
| **CT** [2] | Used to determine the normalization strategy. |

**Note** 
The keywords **above** are **case-sensitive**; a channel is flagged as soon as the keyword appears **anywhere** in its name.



[1] Class names and their corresponding numeric IDs are stored as key–value pairs under `labels` in `dataset.yaml`.

[2] Key–value pairs mapping modality IDs to modality names are stored under `channel_names` in `dataset.yaml`.

