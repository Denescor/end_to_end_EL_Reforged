-- The code in this file does two things:
--   a) extracts and puts the entity relatedness dataset in two maps (reltd_validate and
--      reltd_test). Provides functions to evaluate entity embeddings on this dataset
--      (Table 1 in our paper).
--   b) extracts all entities that appear in any of the ED (as mention candidates) or
--      entity relatedness datasets. These are placed in an object called rewtr that will 
--      be used to restrict the set of entities for which we want to train entity embeddings 
--      (done with the file entities/learn_e2v/learn_a.lua).

if not opt then
  cmd = torch.CmdLine()
  cmd:option('-root_data_dir', '', 'Root path of the data, $DATA_PATH.')
  cmd:option('-input_dir', 'generated/')
  cmd:option('-output_dir', '/people/carpentier/Modeles/end2end_neural_el-master/data/entities/wikiid2nnid/')
  cmd:text()
  opt = cmd:parse(arg or {})
  assert(opt.root_data_dir ~= '', 'Specify a valid root_data_dir path argument.')  
end

dofile '/people/carpentier/Modeles/en_entities/deep-ed-master_true_map/utils/utils.lua'
tds = tds or require 'tds'

--------------------------------------------------------------
------------------------ Main code ---------------------------
--------------------------------------------------------------

local rewtr_t7filename = opt.input_dir .. 'all_candidate_ents_ed_rltd_datasets_RLTD.t7'
print('==> Loading relatedness thid tensor')
print('  ---> from t7 file.')
rewtr = torch.load(rewtr_t7filename)
print('  ---> done. Now write wikiid2nnid')
-- write "wikiid2nnid" from the tds.Hash()
nb_ent2tltd = 0
nb_rltd2ent = 0
num_ents = rewtr.num_rltd_ents
out = assert(io.open(opt.output_dir .. "wikiid2nnid.txt", "w"))
for wikiid, rltd_id in pairs(rewtr.reltd_ents_wikiid_to_rltdid) do
  out:write(wikiid .. "\t" .. rltd_id .. "\n")
  nb_ent2tltd = nb_ent2tltd + 1
end 
out:close()
out = assert(io.open(opt.output_dir .. "nnid2wikiid.txt", "w"))
for rltd_id, wikiid in pairs(rewtr.reltd_ents_rltdid_to_wikiid) do
  out:write(rltd_id .. "\t" .. wikiid .. "\n")
  nb_rltd2ent = nb_rltd2ent + 1
end
out:close()  
print("  ---> done.\nTotal ents : " .. num_ents .. "\nlen wikiid2nnid : " .. nb_ent2tltd .. "\nlen nnid2wikiid : " .. nb_rltd2ent)
