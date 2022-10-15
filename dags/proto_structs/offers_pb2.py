#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0coffers.proto\"\x1f\n\x06Offers\x12\x15\n\x05offer\x18\x01 \x03(\x0b\x32\x06.Offer\"\xbc\x01\n\x05Offer\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x14\n\x0cprice_before\x18\x04 \x01(\x02\x12\x13\n\x0bprice_after\x18\x05 \x01(\x02\x12\x0e\n\x06\x61mount\x18\x08 \x01(\x02\x12\x13\n\x0b\x61mount_unit\x18\t \x01(\t\x12\x10\n\x08\x64iscount\x18\n \x01(\x02\x12\x1d\n\x12\x64iscount_condition\x18\x0c \x01(\t:\x01-\x12\x12\n\nstart_date\x18\x0f \x01(\t\x12\x10\n\x08\x65nd_date\x18\x10 \x01(\t')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'offers_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

    DESCRIPTOR._options = None
    _OFFERS._serialized_start=16
    _OFFERS._serialized_end=47
    _OFFER._serialized_start=50
    _OFFER._serialized_end=238
# @@protoc_insertion_point(module_scope)

