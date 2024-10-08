import uproot
import awkward as ak
import numpy as np
#t["trkstsh"]["field"][i][j] event i track j
#f = uproot.open("file_name.root")
#f.keys()
#This script puts data and mc fields into a dictionary. Each dictionary entry is just a 1d list of all the hits
#TODO: need to put data fields that we care about, which is the data we're indexing in on the RHS
#(for example:detsh.udocavar in udocasig = ak.concatenate(trkana['detsh.udocavar']).to_numpy()) in the dict, and get
#script working
#cut picks out tracks with good status

files = ["/global/cfs/cdirs/m3712/Mu2e/TA_CeEndpointMix1BBTriggerable.MDC2020ae_best_v1_3.root"]

data_fields = ["state", "udoca", "cdrift", "rdrift", "tottdrift", "wdot", "udocavar", "wdist", "uupos", "poca"]
mc_fields = ["rel"]

a = {field: [] for field in data_fields}
for field in mc_fields:
    a[field] = []

for batch in uproot.iterate(files,filter_name="/trk|trktsh|trktshmc/i"):
    cut = ak.sum(batch["trk.status"],axis=1) == 1
    for field in data_fields:
        a[field].append(ak.flatten(ak.flatten(batch["trktsh"][field][cut])).to_numpy())
    for field in mc_fields:
        a[field].append(ak.flatten(ak.flatten(batch["trktshmc"][field][ak.local_index(batch["trktshmc"][field]) < ak.num(batch["trktsh"]
            [data_fields[0]], axis = 2)][cut])).to_numpy())
for field in a:
    a[field] = np.concatenate(a[field])
print (len(a["state"]))
print(len(a["rel"]))
#import pdb;pdb.set_trace()
