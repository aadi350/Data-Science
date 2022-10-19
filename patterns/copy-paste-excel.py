import pandas as pd

# meant to be copy-pasted from an excel directly
schedule_string = """G-VIIO	BAW	LGW	ANU	10:00	13:35
G-YMMF BAW LGW KIN 12:45 4:30
G-VIIX BAW LGW JFK 15:00 18:00
G-NEOZ BAW CAI LHR 16:35 20:55
D-AISN DLH FRA PMI 16:15 18:20
D-AIEB DLH TLV FRA 16:30 20:00
RA-73768 AFL NAL SVO 17:50 21:25
VP-BXA AFL KJA SVO 14:00 20:33
"""

schedule = pd.DataFrame(
    data=[x.split("\t") for x in schedule_string.splitlines()],
    columns=[
        "registration",
        "callsign",
        "depart",
        "arrive",
        "time_depart",
        "time_arrive",
    ],
)
