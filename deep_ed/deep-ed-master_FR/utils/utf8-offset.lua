local bit = require 'bitop.funcs'

local function strRelToAbs( str, ... )
	local args = { ... }
	for k, v in ipairs( args ) do
		v = v > 0 and v or #str + v + 1
		if v < 1 or v > #str then
			error( "bad index to string (out of range)", 3 )
		end
		args[ k ] = v
	end
	return table.unpack(args)   
end

local function decode( str, startPos )
	startPos = strRelToAbs( str, startPos or 1 )
	local b1 = str:byte( startPos, startPos )
	-- Single-byte sequence
	if b1 < 0x80 then
		return startPos, startPos
	end
	-- Validate first byte of multi-byte sequence
	if b1 > 0xF4 or b1 < 0xC2 then
		return nil
	end
	-- Get 'supposed' amount of continuation bytes from primary byte
	local contByteCount =	b1 >= 0xF0 and 3 or
							b1 >= 0xE0 and 2 or
							b1 >= 0xC0 and 1
	local endPos = startPos + contByteCount
	-- Validate our continuation bytes
	for _, bX in ipairs { str:byte( startPos + 1, endPos ) } do
		if bit.band( bX, 0xC0 ) ~= 0x80 then
			return nil
		end
	end
	return startPos, endPos 
end

local function offset( str, n, startPos )
	startPos = strRelToAbs( str, startPos or ( n >= 0 and 1 ) or #str )
	-- Find the beginning of the sequence over startPos
	if n == 0 then
		for i = startPos, 1, -1 do
			local seqStartPos, seqEndPos = decode( str, i )
			if seqStartPos then
				return seqStartPos
			end
		end
		return nil
	end
	if not decode( str, startPos ) then
		error( "initial position is not beginning of a valid sequence", 2 )
	end
	local itStart, itEnd, itStep = nil, nil, nil
	if n > 0 then -- Find the beginning of the n'th sequence forwards
		itStart = startPos
		itEnd = #str
		itStep = 1
	else -- Find the beginning of the n'th sequence backwards
		n = -n
		itStart = startPos
		itEnd = 1
		itStep = -1
	end
	for i = itStart, itEnd, itStep do
		local seqStartPos, seqEndPos = decode( str, i )
		if seqStartPos then
			n = n - 1
			if n == 0 then
				return seqStartPos
			end
		end
	end
	return nil
end

function utf8sub(s,i,j)
    if s == "" or s == nil then
        return s
    else
        i2= offset(s,i)
        if i2 == nil then print("offset begin (" .. i .. ") nil : " .. s) end
        j2= offset(s,j+1)
        if j2 == nil then print("offset end (" .. j .. ") nil : " .. s) else j2 = j2 - 1 end
        return string.sub(s,i2,j2)
    end
end
