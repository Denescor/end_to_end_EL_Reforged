
require 'lfs'
dofile 'utils/utf8-offset.lua'

if not ent_p_e_m_index then
  require 'torch'
  dofile 'data_gen/indexes/wiki_redirects_index.lua'
  dofile 'data_gen/indexes/yago_crosswikis_wiki.lua'
  dofile 'utils/utils.lua'
end

tds = tds or require 'tds'

print('\nGenerating test data from TR2016 train set')

out_file_A = opt.root_data_dir .. 'generated/test_train_data/TR_train.csv'
out_file_B = opt.root_data_dir .. 'generated/test_train_data/TR_test.csv'

in_train = opt.root_data_dir .. '../TR/train/'
in_test = opt.root_data_dir .. '../TR/test/'

ouf_A = assert(io.open(out_file_A, "w"))
ouf_B = assert(io.open(out_file_B, "w"))

local num_nme = 0
local num_nonexistent_ent_title = 0
local num_nonexistent_ent_id = 0
local num_nonexistent_both = 0
local num_correct_ents = 0
local num_total_ents = 0

local cur_words_num = 0
local cur_words = {}
local cur_mentions = {}
local cur_mentions_num = 0

local cur_doc_name = ''
local unwrite = 0
local i_test = 0

function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

local function write_results()
  -- Write results:
  if cur_doc_name ~= '' then
    local header = cur_doc_name .. '\t' .. cur_doc_name .. '\t'
    for _, hyp in pairs(cur_mentions) do
      if hyp.mention:len() == 0 then
        print(hyp.mention)
        print(line)
        assert(hyp.mention:len() > 0, line)
      end
      local mention = string.gsub(hyp.mention, '\t', ' ')
      local str = header .. hyp.mention .. '\t'

      local left_ctxt = {}
      for i = math.max(0, hyp.start_off - 100), hyp.start_off - 1 do
        table.insert(left_ctxt, cur_words[i])
      end
      if table_len(left_ctxt) == 0 then
        table.insert(left_ctxt, 'EMPTYCTXT')
      end
      left_ctxt = string.gsub(table.concat(left_ctxt, ' '), '\n', ' ')
      --left_ctxt = string.gsub(table.concat(left_ctxt, ' '), '\n', ' ')
      str = str .. left_ctxt .. '\t'

      local right_ctxt = {}
      for i = hyp.end_off + 1, math.min(cur_words_num, hyp.end_off + 100) do
        table.insert(right_ctxt, cur_words[i])
      end
      if table_len(right_ctxt) == 0 then
        table.insert(right_ctxt, 'EMPTYCTXT')
      end
      right_ctxt = string.gsub(table.concat(right_ctxt, ' '), '\n', ' ')
      --right_ctxt = string.gsub(table.concat(right_ctxt, ' '), '\n', ' ')
      str = str .. right_ctxt .. '\tCANDIDATES\t'

      -- Entity candidates from p(e|m) dictionary
      if ent_p_e_m_index[mention] and #(ent_p_e_m_index[mention]) > 0 then

        local sorted_cand = {}
        for ent_wikiid,p in pairs(ent_p_e_m_index[mention]) do
          table.insert(sorted_cand, {ent_wikiid = ent_wikiid, p = p})
        end
        table.sort(sorted_cand, function(a,b) return a.p > b.p end)

        local candidates = {}
        local gt_pos = -1
        for pos,e in pairs(sorted_cand) do
          if pos <= 100 then
            table.insert(candidates, e.ent_wikiid .. ',' .. string.format("%.3f", e.p) .. ',' .. get_ent_name_from_wikiid(e.ent_wikiid))
            if e.ent_wikiid == hyp.ent_wikiid then
              gt_pos = pos
            end
          else
            break
          end
        end
        str = str .. table.concat(candidates, '\t') .. '\tGT:\t'

        if gt_pos > 0 then
          ouf:write(str .. gt_pos .. ',' .. candidates[gt_pos] .. '\n')
        else
          if hyp.ent_wikiid ~= unk_ent_wikiid then
            ouf:write(str .. '-1,' .. hyp.ent_wikiid .. ',' .. get_ent_name_from_wikiid(hyp.ent_wikiid) .. '\n')
          else
            ouf:write(str .. '-1\n')
          end
        end
      else
        unwrite = unwrite + 1
        if hyp.ent_wikiid ~= unk_ent_wikiid then
          ouf:write(str .. 'EMPTYCAND\tGT:\t-1,' .. hyp.ent_wikiid .. ',' .. get_ent_name_from_wikiid(hyp.ent_wikiid) .. '\n')
        else
          ouf:write(str .. 'EMPTYCAND\tGT:\t-1\n')
        end
      end
    end
  else
    unwrite = unwrite + 0
  end
end

local function extract_TR(str_file,mention_file)
  it, _ = io.open(str_file)
  ref,_ = io.open(mention_file)
  local line = it:read()
  local final_file = ""
  local nb_lines = 0
  cur_words_num = 0
  cur_words = {}
  cur_mentions = {}
  cur_mentions_num = 0
  while (line) do
    final_file = final_file .. line .. '\n'
    nb_lines = nb_lines + 1
    line = it:read()
  end
  line = ref:read()
  while (line) do
    local parts = split(line, '\t')
    local bg_mention = tonumber(parts[1])
    local nd_mention = tonumber(parts[2])
    local en_ent = parts[3]
    local fr_ent = parts[4]
    local tr_ent = fr_ent

    local index_ent_wikiid = get_ent_wikiid_from_name(tr_ent)
    local index_ent_name = get_ent_name_from_wikiid(index_ent_wikiid)
    local cur_mention = utf8sub(final_file,bg_mention,nd_mention)
    if cur_mention:len() > 0 then
        if index_ent_name ~= en_ent then
          num_nonexistent_ent_title = num_nonexistent_ent_title + 1
        else
          num_correct_ents = num_correct_ents + 1
        end
        
        if i_test % 5000 == 0 then
          print('log current entities')
          print('final_file (' .. str_file .. ' ; ' .. nb_lines .. ' lines) : ####\n' .. final_file .. '\n#################')
          print('ent (' .. bg_mention .. ', ' .. nd_mention .. ') : ' .. tr_ent)
          print('mention : ' .. cur_mention)
          print(parts)
        end
        i_test = i_test + 1
        -- TODO trouver comment déterminer les inexistants et introuvable ("num_nonexistent_ent_title" & "num_nonexistent_ent_id")
    
        local final_ent_wikiid = index_ent_wikiid
        --if final_ent_wikiid == unk_ent_wikiid then
        --  final_ent_wikiid = cur_ent_wikiid
        --end
        
        num_nme = num_nme + 1
    
        num_total_ents = num_total_ents + 1 -- Keep even incorrect links
        
        cur_mentions_num = cur_mentions_num + 1
        cur_mentions[cur_mentions_num] = {}
        cur_mentions[cur_mentions_num].mention = cur_mention
        cur_mentions[cur_mentions_num].ent_wikiid = final_ent_wikiid
        cur_mentions[cur_mentions_num].start_off = bg_mention
        cur_mentions[cur_mentions_num].end_off = nd_mention
    
        local words_on_this_line = split_in_words(final_file)
        for _,w in pairs(words_on_this_line) do
          table.insert(cur_words, modify_uppercase_phrase(w))
          cur_words_num =  cur_words_num + 1
        end
    else
      num_nonexistent_both = num_nonexistent_both + 1
    end
    line = ref:read()
  end
  -- print('file : ' .. cur_doc_name .. ' -- ' .. cur_mentions_num .. ' mentions')
  write_results()
end

local function extract_files(folder)
  print(folder)
  local nb_files = 0
  local nb_extract = 0
  list_file = {}
  for file in lfs.dir(folder) do
    if lfs.attributes(file,"mode") ~= "directory" then
      local parts_file = split(file, '.')
      local len_parts = table_len(parts_file)
      local name_file = table.concat(subrange(parts_file,1,len_parts-1),".")
      if list_file[name_file] == nil then
        list_file[name_file] = {}
      end
      table.insert(list_file[name_file],file)
      nb_files = nb_files + 1
    --else
      --mode = lfs.attributes(file,"mode")
      --if mode ~= nil then
      --  print(file .. ' --> ' .. mode)
      --else
      --  print(file .. ' --> nil')
      --end
    end
  end
  print('files to extract : ' .. nb_files)
  for name,files in pairs(list_file) do
    local parts_file = split(files[1], '.')
    local len_parts = table_len(parts_file)
    if parts_file[len_parts] == "txt" then
      file_t = files[1]
      file_m = files[2]
    else
      file_t = files[2]
      file_m = files[1]
    end
    cur_doc_name = name
    if file_t == nil then
      print("bad TR file :" .. file_m)
    end
    if file_m == nil then
      print("bad TR file :" .. file_t)
    else
      extract_TR(folder .. file_t, folder .. file_m)
      nb_extract = nb_extract + 1
    end
  end
  print('files extracted : ' .. nb_extract)
end

print('extract TR train')
ouf = ouf_A
extract_files(in_train)
ouf_A:flush()
io.close(ouf_A)
print('extract TR test')
ouf = ouf_B
extract_files(in_test)
ouf_B:flush()
io.close(ouf_B)
print('no candidates : ' .. unwrite)

print('    Done TR.')
print('num_nme = ' .. num_nme .. '; num_nonexistent_ent_title = ' .. num_nonexistent_ent_title)
print('num_nonexistent_ent_id = ' .. num_nonexistent_ent_id .. '; empty mention = ' .. num_nonexistent_both)
print('num_correct_ents = ' .. num_correct_ents .. '; num_total_ents = ' .. num_total_ents)
