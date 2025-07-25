��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028��
�
2sequential_1/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *C

debug_name53sequential_1/batch_normalization_3/moving_variance/*
dtype0*
shape:�*C
shared_name42sequential_1/batch_normalization_3/moving_variance
�
Fsequential_1/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_1/batch_normalization_3/moving_variance*
_output_shapes	
:�*
dtype0
�
2sequential_1/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *C

debug_name53sequential_1/batch_normalization_2/moving_variance/*
dtype0*
shape:�*C
shared_name42sequential_1/batch_normalization_2/moving_variance
�
Fsequential_1/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_1/batch_normalization_2/moving_variance*
_output_shapes	
:�*
dtype0
�
0sequential_1/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *A

debug_name31sequential_1/batch_normalization/moving_variance/*
dtype0*
shape: *A
shared_name20sequential_1/batch_normalization/moving_variance
�
Dsequential_1/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp0sequential_1/batch_normalization/moving_variance*
_output_shapes
: *
dtype0
�
.sequential_1/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *?

debug_name1/sequential_1/batch_normalization_4/moving_mean/*
dtype0*
shape:�*?
shared_name0.sequential_1/batch_normalization_4/moving_mean
�
Bsequential_1/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_1/batch_normalization_4/moving_mean*
_output_shapes	
:�*
dtype0
�
.sequential_1/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *?

debug_name1/sequential_1/batch_normalization_2/moving_mean/*
dtype0*
shape:�*?
shared_name0.sequential_1/batch_normalization_2/moving_mean
�
Bsequential_1/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_1/batch_normalization_2/moving_mean*
_output_shapes	
:�*
dtype0
�
2sequential_1/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *C

debug_name53sequential_1/batch_normalization_1/moving_variance/*
dtype0*
shape:@*C
shared_name42sequential_1/batch_normalization_1/moving_variance
�
Fsequential_1/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_1/batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
�
,sequential_1/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *=

debug_name/-sequential_1/batch_normalization/moving_mean/*
dtype0*
shape: *=
shared_name.,sequential_1/batch_normalization/moving_mean
�
@sequential_1/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp,sequential_1/batch_normalization/moving_mean*
_output_shapes
: *
dtype0
�
.sequential_1/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *?

debug_name1/sequential_1/batch_normalization_1/moving_mean/*
dtype0*
shape:@*?
shared_name0.sequential_1/batch_normalization_1/moving_mean
�
Bsequential_1/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_1/batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
�
2sequential_1/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *C

debug_name53sequential_1/batch_normalization_4/moving_variance/*
dtype0*
shape:�*C
shared_name42sequential_1/batch_normalization_4/moving_variance
�
Fsequential_1/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_1/batch_normalization_4/moving_variance*
_output_shapes	
:�*
dtype0
�
.sequential_1/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *?

debug_name1/sequential_1/batch_normalization_3/moving_mean/*
dtype0*
shape:�*?
shared_name0.sequential_1/batch_normalization_3/moving_mean
�
Bsequential_1/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_1/batch_normalization_3/moving_mean*
_output_shapes	
:�*
dtype0
�
sequential_1/dense/kernelVarHandleOp*
_output_shapes
: **

debug_namesequential_1/dense/kernel/*
dtype0*
shape:
�b�**
shared_namesequential_1/dense/kernel
�
-sequential_1/dense/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense/kernel* 
_output_shapes
:
�b�*
dtype0
�
sequential_1/conv2d_4/biasVarHandleOp*
_output_shapes
: *+

debug_namesequential_1/conv2d_4/bias/*
dtype0*
shape:�*+
shared_namesequential_1/conv2d_4/bias
�
.sequential_1/conv2d_4/bias/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_4/bias*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_3/biasVarHandleOp*
_output_shapes
: *+

debug_namesequential_1/conv2d_3/bias/*
dtype0*
shape:�*+
shared_namesequential_1/conv2d_3/bias
�
.sequential_1/conv2d_3/bias/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_3/bias*
_output_shapes	
:�*
dtype0
�
(sequential_1/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *9

debug_name+)sequential_1/batch_normalization_2/gamma/*
dtype0*
shape:�*9
shared_name*(sequential_1/batch_normalization_2/gamma
�
<sequential_1/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp(sequential_1/batch_normalization_2/gamma*
_output_shapes	
:�*
dtype0
�
'sequential_1/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *8

debug_name*(sequential_1/batch_normalization_1/beta/*
dtype0*
shape:@*8
shared_name)'sequential_1/batch_normalization_1/beta
�
;sequential_1/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp'sequential_1/batch_normalization_1/beta*
_output_shapes
:@*
dtype0
�
sequential_1/conv2d/biasVarHandleOp*
_output_shapes
: *)

debug_namesequential_1/conv2d/bias/*
dtype0*
shape: *)
shared_namesequential_1/conv2d/bias
�
,sequential_1/conv2d/bias/Read/ReadVariableOpReadVariableOpsequential_1/conv2d/bias*
_output_shapes
: *
dtype0
�
'sequential_1/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *8

debug_name*(sequential_1/batch_normalization_3/beta/*
dtype0*
shape:�*8
shared_name)'sequential_1/batch_normalization_3/beta
�
;sequential_1/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp'sequential_1/batch_normalization_3/beta*
_output_shapes	
:�*
dtype0
�
&sequential_1/batch_normalization/gammaVarHandleOp*
_output_shapes
: *7

debug_name)'sequential_1/batch_normalization/gamma/*
dtype0*
shape: *7
shared_name(&sequential_1/batch_normalization/gamma
�
:sequential_1/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp&sequential_1/batch_normalization/gamma*
_output_shapes
: *
dtype0
�
sequential_1/dense/biasVarHandleOp*
_output_shapes
: *(

debug_namesequential_1/dense/bias/*
dtype0*
shape:�*(
shared_namesequential_1/dense/bias
�
+sequential_1/dense/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense/bias*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_2/kernelVarHandleOp*
_output_shapes
: *-

debug_namesequential_1/conv2d_2/kernel/*
dtype0*
shape:@�*-
shared_namesequential_1/conv2d_2/kernel
�
0sequential_1/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_2/kernel*'
_output_shapes
:@�*
dtype0
�
sequential_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *+

debug_namesequential_1/conv2d_1/bias/*
dtype0*
shape:@*+
shared_namesequential_1/conv2d_1/bias
�
.sequential_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_1/bias*
_output_shapes
:@*
dtype0
�
sequential_1/dense_1/kernelVarHandleOp*
_output_shapes
: *,

debug_namesequential_1/dense_1/kernel/*
dtype0*
shape:	�*,
shared_namesequential_1/dense_1/kernel
�
/sequential_1/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential_1/dense_1/kernel*
_output_shapes
:	�*
dtype0
�
(sequential_1/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *9

debug_name+)sequential_1/batch_normalization_4/gamma/*
dtype0*
shape:�*9
shared_name*(sequential_1/batch_normalization_4/gamma
�
<sequential_1/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp(sequential_1/batch_normalization_4/gamma*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_4/kernelVarHandleOp*
_output_shapes
: *-

debug_namesequential_1/conv2d_4/kernel/*
dtype0*
shape:��*-
shared_namesequential_1/conv2d_4/kernel
�
0sequential_1/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_4/kernel*(
_output_shapes
:��*
dtype0
�
sequential_1/conv2d_3/kernelVarHandleOp*
_output_shapes
: *-

debug_namesequential_1/conv2d_3/kernel/*
dtype0*
shape:��*-
shared_namesequential_1/conv2d_3/kernel
�
0sequential_1/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_3/kernel*(
_output_shapes
:��*
dtype0
�
sequential_1/dense_1/biasVarHandleOp*
_output_shapes
: **

debug_namesequential_1/dense_1/bias/*
dtype0*
shape:**
shared_namesequential_1/dense_1/bias
�
-sequential_1/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential_1/dense_1/bias*
_output_shapes
:*
dtype0
�
(sequential_1/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *9

debug_name+)sequential_1/batch_normalization_1/gamma/*
dtype0*
shape:@*9
shared_name*(sequential_1/batch_normalization_1/gamma
�
<sequential_1/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp(sequential_1/batch_normalization_1/gamma*
_output_shapes
:@*
dtype0
�
'sequential_1/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *8

debug_name*(sequential_1/batch_normalization_4/beta/*
dtype0*
shape:�*8
shared_name)'sequential_1/batch_normalization_4/beta
�
;sequential_1/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp'sequential_1/batch_normalization_4/beta*
_output_shapes	
:�*
dtype0
�
(sequential_1/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *9

debug_name+)sequential_1/batch_normalization_3/gamma/*
dtype0*
shape:�*9
shared_name*(sequential_1/batch_normalization_3/gamma
�
<sequential_1/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp(sequential_1/batch_normalization_3/gamma*
_output_shapes	
:�*
dtype0
�
'sequential_1/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *8

debug_name*(sequential_1/batch_normalization_2/beta/*
dtype0*
shape:�*8
shared_name)'sequential_1/batch_normalization_2/beta
�
;sequential_1/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp'sequential_1/batch_normalization_2/beta*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_2/biasVarHandleOp*
_output_shapes
: *+

debug_namesequential_1/conv2d_2/bias/*
dtype0*
shape:�*+
shared_namesequential_1/conv2d_2/bias
�
.sequential_1/conv2d_2/bias/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_2/bias*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *-

debug_namesequential_1/conv2d_1/kernel/*
dtype0*
shape: @*-
shared_namesequential_1/conv2d_1/kernel
�
0sequential_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
�
sequential_1/conv2d/kernelVarHandleOp*
_output_shapes
: *+

debug_namesequential_1/conv2d/kernel/*
dtype0*
shape: *+
shared_namesequential_1/conv2d/kernel
�
.sequential_1/conv2d/kernel/Read/ReadVariableOpReadVariableOpsequential_1/conv2d/kernel*&
_output_shapes
: *
dtype0
�
%sequential_1/batch_normalization/betaVarHandleOp*
_output_shapes
: *6

debug_name(&sequential_1/batch_normalization/beta/*
dtype0*
shape: *6
shared_name'%sequential_1/batch_normalization/beta
�
9sequential_1/batch_normalization/beta/Read/ReadVariableOpReadVariableOp%sequential_1/batch_normalization/beta*
_output_shapes
: *
dtype0
�
sequential_1/dense_1/bias_1VarHandleOp*
_output_shapes
: *,

debug_namesequential_1/dense_1/bias_1/*
dtype0*
shape:*,
shared_namesequential_1/dense_1/bias_1
�
/sequential_1/dense_1/bias_1/Read/ReadVariableOpReadVariableOpsequential_1/dense_1/bias_1*
_output_shapes
:*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpsequential_1/dense_1/bias_1*
_class
loc:@Variable*
_output_shapes
:*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0
�
sequential_1/dense_1/kernel_1VarHandleOp*
_output_shapes
: *.

debug_name sequential_1/dense_1/kernel_1/*
dtype0*
shape:	�*.
shared_namesequential_1/dense_1/kernel_1
�
1sequential_1/dense_1/kernel_1/Read/ReadVariableOpReadVariableOpsequential_1/dense_1/kernel_1*
_output_shapes
:	�*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpsequential_1/dense_1/kernel_1*
_class
loc:@Variable_1*
_output_shapes
:	�*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape:	�*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	�*
dtype0
�
%seed_generator_3/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_3/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_3/seed_generator_state
�
9seed_generator_3/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_3/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_2/Initializer/ReadVariableOpReadVariableOp%seed_generator_3/seed_generator_state*
_class
loc:@Variable_2*
_output_shapes
:*
dtype0	
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0	*
shape:*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0	
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
�
sequential_1/dense/bias_1VarHandleOp*
_output_shapes
: **

debug_namesequential_1/dense/bias_1/*
dtype0*
shape:�**
shared_namesequential_1/dense/bias_1
�
-sequential_1/dense/bias_1/Read/ReadVariableOpReadVariableOpsequential_1/dense/bias_1*
_output_shapes	
:�*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpsequential_1/dense/bias_1*
_class
loc:@Variable_3*
_output_shapes	
:�*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:�*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
f
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes	
:�*
dtype0
�
sequential_1/dense/kernel_1VarHandleOp*
_output_shapes
: *,

debug_namesequential_1/dense/kernel_1/*
dtype0*
shape:
�b�*,
shared_namesequential_1/dense/kernel_1
�
/sequential_1/dense/kernel_1/Read/ReadVariableOpReadVariableOpsequential_1/dense/kernel_1* 
_output_shapes
:
�b�*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpsequential_1/dense/kernel_1*
_class
loc:@Variable_4* 
_output_shapes
:
�b�*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:
�b�*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
k
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4* 
_output_shapes
:
�b�*
dtype0
�
4sequential_1/batch_normalization_4/moving_variance_1VarHandleOp*
_output_shapes
: *E

debug_name75sequential_1/batch_normalization_4/moving_variance_1/*
dtype0*
shape:�*E
shared_name64sequential_1/batch_normalization_4/moving_variance_1
�
Hsequential_1/batch_normalization_4/moving_variance_1/Read/ReadVariableOpReadVariableOp4sequential_1/batch_normalization_4/moving_variance_1*
_output_shapes	
:�*
dtype0
�
%Variable_5/Initializer/ReadVariableOpReadVariableOp4sequential_1/batch_normalization_4/moving_variance_1*
_class
loc:@Variable_5*
_output_shapes	
:�*
dtype0
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0*
shape:�*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0
f
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes	
:�*
dtype0
�
0sequential_1/batch_normalization_4/moving_mean_1VarHandleOp*
_output_shapes
: *A

debug_name31sequential_1/batch_normalization_4/moving_mean_1/*
dtype0*
shape:�*A
shared_name20sequential_1/batch_normalization_4/moving_mean_1
�
Dsequential_1/batch_normalization_4/moving_mean_1/Read/ReadVariableOpReadVariableOp0sequential_1/batch_normalization_4/moving_mean_1*
_output_shapes	
:�*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOp0sequential_1/batch_normalization_4/moving_mean_1*
_class
loc:@Variable_6*
_output_shapes	
:�*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:�*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
f
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes	
:�*
dtype0
�
)sequential_1/batch_normalization_4/beta_1VarHandleOp*
_output_shapes
: *:

debug_name,*sequential_1/batch_normalization_4/beta_1/*
dtype0*
shape:�*:
shared_name+)sequential_1/batch_normalization_4/beta_1
�
=sequential_1/batch_normalization_4/beta_1/Read/ReadVariableOpReadVariableOp)sequential_1/batch_normalization_4/beta_1*
_output_shapes	
:�*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOp)sequential_1/batch_normalization_4/beta_1*
_class
loc:@Variable_7*
_output_shapes	
:�*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:�*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
f
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7*
_output_shapes	
:�*
dtype0
�
*sequential_1/batch_normalization_4/gamma_1VarHandleOp*
_output_shapes
: *;

debug_name-+sequential_1/batch_normalization_4/gamma_1/*
dtype0*
shape:�*;
shared_name,*sequential_1/batch_normalization_4/gamma_1
�
>sequential_1/batch_normalization_4/gamma_1/Read/ReadVariableOpReadVariableOp*sequential_1/batch_normalization_4/gamma_1*
_output_shapes	
:�*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOp*sequential_1/batch_normalization_4/gamma_1*
_class
loc:@Variable_8*
_output_shapes	
:�*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:�*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
f
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_4/bias_1VarHandleOp*
_output_shapes
: *-

debug_namesequential_1/conv2d_4/bias_1/*
dtype0*
shape:�*-
shared_namesequential_1/conv2d_4/bias_1
�
0sequential_1/conv2d_4/bias_1/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_4/bias_1*
_output_shapes	
:�*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOpsequential_1/conv2d_4/bias_1*
_class
loc:@Variable_9*
_output_shapes	
:�*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape:�*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
f
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_4/kernel_1VarHandleOp*
_output_shapes
: */

debug_name!sequential_1/conv2d_4/kernel_1/*
dtype0*
shape:��*/
shared_name sequential_1/conv2d_4/kernel_1
�
2sequential_1/conv2d_4/kernel_1/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_4/kernel_1*(
_output_shapes
:��*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOpsequential_1/conv2d_4/kernel_1*
_class
loc:@Variable_10*(
_output_shapes
:��*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape:��*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
u
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*(
_output_shapes
:��*
dtype0
�
4sequential_1/batch_normalization_3/moving_variance_1VarHandleOp*
_output_shapes
: *E

debug_name75sequential_1/batch_normalization_3/moving_variance_1/*
dtype0*
shape:�*E
shared_name64sequential_1/batch_normalization_3/moving_variance_1
�
Hsequential_1/batch_normalization_3/moving_variance_1/Read/ReadVariableOpReadVariableOp4sequential_1/batch_normalization_3/moving_variance_1*
_output_shapes	
:�*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOp4sequential_1/batch_normalization_3/moving_variance_1*
_class
loc:@Variable_11*
_output_shapes	
:�*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape:�*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
h
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*
_output_shapes	
:�*
dtype0
�
0sequential_1/batch_normalization_3/moving_mean_1VarHandleOp*
_output_shapes
: *A

debug_name31sequential_1/batch_normalization_3/moving_mean_1/*
dtype0*
shape:�*A
shared_name20sequential_1/batch_normalization_3/moving_mean_1
�
Dsequential_1/batch_normalization_3/moving_mean_1/Read/ReadVariableOpReadVariableOp0sequential_1/batch_normalization_3/moving_mean_1*
_output_shapes	
:�*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOp0sequential_1/batch_normalization_3/moving_mean_1*
_class
loc:@Variable_12*
_output_shapes	
:�*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:�*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
h
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes	
:�*
dtype0
�
)sequential_1/batch_normalization_3/beta_1VarHandleOp*
_output_shapes
: *:

debug_name,*sequential_1/batch_normalization_3/beta_1/*
dtype0*
shape:�*:
shared_name+)sequential_1/batch_normalization_3/beta_1
�
=sequential_1/batch_normalization_3/beta_1/Read/ReadVariableOpReadVariableOp)sequential_1/batch_normalization_3/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOp)sequential_1/batch_normalization_3/beta_1*
_class
loc:@Variable_13*
_output_shapes	
:�*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:�*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
h
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*
_output_shapes	
:�*
dtype0
�
*sequential_1/batch_normalization_3/gamma_1VarHandleOp*
_output_shapes
: *;

debug_name-+sequential_1/batch_normalization_3/gamma_1/*
dtype0*
shape:�*;
shared_name,*sequential_1/batch_normalization_3/gamma_1
�
>sequential_1/batch_normalization_3/gamma_1/Read/ReadVariableOpReadVariableOp*sequential_1/batch_normalization_3/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOp*sequential_1/batch_normalization_3/gamma_1*
_class
loc:@Variable_14*
_output_shapes	
:�*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:�*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
h
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_3/bias_1VarHandleOp*
_output_shapes
: *-

debug_namesequential_1/conv2d_3/bias_1/*
dtype0*
shape:�*-
shared_namesequential_1/conv2d_3/bias_1
�
0sequential_1/conv2d_3/bias_1/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_3/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOpsequential_1/conv2d_3/bias_1*
_class
loc:@Variable_15*
_output_shapes	
:�*
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape:�*
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
h
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_3/kernel_1VarHandleOp*
_output_shapes
: */

debug_name!sequential_1/conv2d_3/kernel_1/*
dtype0*
shape:��*/
shared_name sequential_1/conv2d_3/kernel_1
�
2sequential_1/conv2d_3/kernel_1/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_3/kernel_1*(
_output_shapes
:��*
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOpsequential_1/conv2d_3/kernel_1*
_class
loc:@Variable_16*(
_output_shapes
:��*
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape:��*
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
u
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*(
_output_shapes
:��*
dtype0
�
4sequential_1/batch_normalization_2/moving_variance_1VarHandleOp*
_output_shapes
: *E

debug_name75sequential_1/batch_normalization_2/moving_variance_1/*
dtype0*
shape:�*E
shared_name64sequential_1/batch_normalization_2/moving_variance_1
�
Hsequential_1/batch_normalization_2/moving_variance_1/Read/ReadVariableOpReadVariableOp4sequential_1/batch_normalization_2/moving_variance_1*
_output_shapes	
:�*
dtype0
�
&Variable_17/Initializer/ReadVariableOpReadVariableOp4sequential_1/batch_normalization_2/moving_variance_1*
_class
loc:@Variable_17*
_output_shapes	
:�*
dtype0
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0*
shape:�*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0
h
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17*
_output_shapes	
:�*
dtype0
�
0sequential_1/batch_normalization_2/moving_mean_1VarHandleOp*
_output_shapes
: *A

debug_name31sequential_1/batch_normalization_2/moving_mean_1/*
dtype0*
shape:�*A
shared_name20sequential_1/batch_normalization_2/moving_mean_1
�
Dsequential_1/batch_normalization_2/moving_mean_1/Read/ReadVariableOpReadVariableOp0sequential_1/batch_normalization_2/moving_mean_1*
_output_shapes	
:�*
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOp0sequential_1/batch_normalization_2/moving_mean_1*
_class
loc:@Variable_18*
_output_shapes	
:�*
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape:�*
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
h
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*
_output_shapes	
:�*
dtype0
�
)sequential_1/batch_normalization_2/beta_1VarHandleOp*
_output_shapes
: *:

debug_name,*sequential_1/batch_normalization_2/beta_1/*
dtype0*
shape:�*:
shared_name+)sequential_1/batch_normalization_2/beta_1
�
=sequential_1/batch_normalization_2/beta_1/Read/ReadVariableOpReadVariableOp)sequential_1/batch_normalization_2/beta_1*
_output_shapes	
:�*
dtype0
�
&Variable_19/Initializer/ReadVariableOpReadVariableOp)sequential_1/batch_normalization_2/beta_1*
_class
loc:@Variable_19*
_output_shapes	
:�*
dtype0
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0*
shape:�*
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0
h
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*
_output_shapes	
:�*
dtype0
�
*sequential_1/batch_normalization_2/gamma_1VarHandleOp*
_output_shapes
: *;

debug_name-+sequential_1/batch_normalization_2/gamma_1/*
dtype0*
shape:�*;
shared_name,*sequential_1/batch_normalization_2/gamma_1
�
>sequential_1/batch_normalization_2/gamma_1/Read/ReadVariableOpReadVariableOp*sequential_1/batch_normalization_2/gamma_1*
_output_shapes	
:�*
dtype0
�
&Variable_20/Initializer/ReadVariableOpReadVariableOp*sequential_1/batch_normalization_2/gamma_1*
_class
loc:@Variable_20*
_output_shapes	
:�*
dtype0
�
Variable_20VarHandleOp*
_class
loc:@Variable_20*
_output_shapes
: *

debug_nameVariable_20/*
dtype0*
shape:�*
shared_nameVariable_20
g
,Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_20*
_output_shapes
: 
h
Variable_20/AssignAssignVariableOpVariable_20&Variable_20/Initializer/ReadVariableOp*
dtype0
h
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_2/bias_1VarHandleOp*
_output_shapes
: *-

debug_namesequential_1/conv2d_2/bias_1/*
dtype0*
shape:�*-
shared_namesequential_1/conv2d_2/bias_1
�
0sequential_1/conv2d_2/bias_1/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_2/bias_1*
_output_shapes	
:�*
dtype0
�
&Variable_21/Initializer/ReadVariableOpReadVariableOpsequential_1/conv2d_2/bias_1*
_class
loc:@Variable_21*
_output_shapes	
:�*
dtype0
�
Variable_21VarHandleOp*
_class
loc:@Variable_21*
_output_shapes
: *

debug_nameVariable_21/*
dtype0*
shape:�*
shared_nameVariable_21
g
,Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_21*
_output_shapes
: 
h
Variable_21/AssignAssignVariableOpVariable_21&Variable_21/Initializer/ReadVariableOp*
dtype0
h
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21*
_output_shapes	
:�*
dtype0
�
sequential_1/conv2d_2/kernel_1VarHandleOp*
_output_shapes
: */

debug_name!sequential_1/conv2d_2/kernel_1/*
dtype0*
shape:@�*/
shared_name sequential_1/conv2d_2/kernel_1
�
2sequential_1/conv2d_2/kernel_1/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_2/kernel_1*'
_output_shapes
:@�*
dtype0
�
&Variable_22/Initializer/ReadVariableOpReadVariableOpsequential_1/conv2d_2/kernel_1*
_class
loc:@Variable_22*'
_output_shapes
:@�*
dtype0
�
Variable_22VarHandleOp*
_class
loc:@Variable_22*
_output_shapes
: *

debug_nameVariable_22/*
dtype0*
shape:@�*
shared_nameVariable_22
g
,Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_22*
_output_shapes
: 
h
Variable_22/AssignAssignVariableOpVariable_22&Variable_22/Initializer/ReadVariableOp*
dtype0
t
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*'
_output_shapes
:@�*
dtype0
�
4sequential_1/batch_normalization_1/moving_variance_1VarHandleOp*
_output_shapes
: *E

debug_name75sequential_1/batch_normalization_1/moving_variance_1/*
dtype0*
shape:@*E
shared_name64sequential_1/batch_normalization_1/moving_variance_1
�
Hsequential_1/batch_normalization_1/moving_variance_1/Read/ReadVariableOpReadVariableOp4sequential_1/batch_normalization_1/moving_variance_1*
_output_shapes
:@*
dtype0
�
&Variable_23/Initializer/ReadVariableOpReadVariableOp4sequential_1/batch_normalization_1/moving_variance_1*
_class
loc:@Variable_23*
_output_shapes
:@*
dtype0
�
Variable_23VarHandleOp*
_class
loc:@Variable_23*
_output_shapes
: *

debug_nameVariable_23/*
dtype0*
shape:@*
shared_nameVariable_23
g
,Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_23*
_output_shapes
: 
h
Variable_23/AssignAssignVariableOpVariable_23&Variable_23/Initializer/ReadVariableOp*
dtype0
g
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23*
_output_shapes
:@*
dtype0
�
0sequential_1/batch_normalization_1/moving_mean_1VarHandleOp*
_output_shapes
: *A

debug_name31sequential_1/batch_normalization_1/moving_mean_1/*
dtype0*
shape:@*A
shared_name20sequential_1/batch_normalization_1/moving_mean_1
�
Dsequential_1/batch_normalization_1/moving_mean_1/Read/ReadVariableOpReadVariableOp0sequential_1/batch_normalization_1/moving_mean_1*
_output_shapes
:@*
dtype0
�
&Variable_24/Initializer/ReadVariableOpReadVariableOp0sequential_1/batch_normalization_1/moving_mean_1*
_class
loc:@Variable_24*
_output_shapes
:@*
dtype0
�
Variable_24VarHandleOp*
_class
loc:@Variable_24*
_output_shapes
: *

debug_nameVariable_24/*
dtype0*
shape:@*
shared_nameVariable_24
g
,Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_24*
_output_shapes
: 
h
Variable_24/AssignAssignVariableOpVariable_24&Variable_24/Initializer/ReadVariableOp*
dtype0
g
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24*
_output_shapes
:@*
dtype0
�
)sequential_1/batch_normalization_1/beta_1VarHandleOp*
_output_shapes
: *:

debug_name,*sequential_1/batch_normalization_1/beta_1/*
dtype0*
shape:@*:
shared_name+)sequential_1/batch_normalization_1/beta_1
�
=sequential_1/batch_normalization_1/beta_1/Read/ReadVariableOpReadVariableOp)sequential_1/batch_normalization_1/beta_1*
_output_shapes
:@*
dtype0
�
&Variable_25/Initializer/ReadVariableOpReadVariableOp)sequential_1/batch_normalization_1/beta_1*
_class
loc:@Variable_25*
_output_shapes
:@*
dtype0
�
Variable_25VarHandleOp*
_class
loc:@Variable_25*
_output_shapes
: *

debug_nameVariable_25/*
dtype0*
shape:@*
shared_nameVariable_25
g
,Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_25*
_output_shapes
: 
h
Variable_25/AssignAssignVariableOpVariable_25&Variable_25/Initializer/ReadVariableOp*
dtype0
g
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25*
_output_shapes
:@*
dtype0
�
*sequential_1/batch_normalization_1/gamma_1VarHandleOp*
_output_shapes
: *;

debug_name-+sequential_1/batch_normalization_1/gamma_1/*
dtype0*
shape:@*;
shared_name,*sequential_1/batch_normalization_1/gamma_1
�
>sequential_1/batch_normalization_1/gamma_1/Read/ReadVariableOpReadVariableOp*sequential_1/batch_normalization_1/gamma_1*
_output_shapes
:@*
dtype0
�
&Variable_26/Initializer/ReadVariableOpReadVariableOp*sequential_1/batch_normalization_1/gamma_1*
_class
loc:@Variable_26*
_output_shapes
:@*
dtype0
�
Variable_26VarHandleOp*
_class
loc:@Variable_26*
_output_shapes
: *

debug_nameVariable_26/*
dtype0*
shape:@*
shared_nameVariable_26
g
,Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_26*
_output_shapes
: 
h
Variable_26/AssignAssignVariableOpVariable_26&Variable_26/Initializer/ReadVariableOp*
dtype0
g
Variable_26/Read/ReadVariableOpReadVariableOpVariable_26*
_output_shapes
:@*
dtype0
�
sequential_1/conv2d_1/bias_1VarHandleOp*
_output_shapes
: *-

debug_namesequential_1/conv2d_1/bias_1/*
dtype0*
shape:@*-
shared_namesequential_1/conv2d_1/bias_1
�
0sequential_1/conv2d_1/bias_1/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_1/bias_1*
_output_shapes
:@*
dtype0
�
&Variable_27/Initializer/ReadVariableOpReadVariableOpsequential_1/conv2d_1/bias_1*
_class
loc:@Variable_27*
_output_shapes
:@*
dtype0
�
Variable_27VarHandleOp*
_class
loc:@Variable_27*
_output_shapes
: *

debug_nameVariable_27/*
dtype0*
shape:@*
shared_nameVariable_27
g
,Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_27*
_output_shapes
: 
h
Variable_27/AssignAssignVariableOpVariable_27&Variable_27/Initializer/ReadVariableOp*
dtype0
g
Variable_27/Read/ReadVariableOpReadVariableOpVariable_27*
_output_shapes
:@*
dtype0
�
sequential_1/conv2d_1/kernel_1VarHandleOp*
_output_shapes
: */

debug_name!sequential_1/conv2d_1/kernel_1/*
dtype0*
shape: @*/
shared_name sequential_1/conv2d_1/kernel_1
�
2sequential_1/conv2d_1/kernel_1/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_1/kernel_1*&
_output_shapes
: @*
dtype0
�
&Variable_28/Initializer/ReadVariableOpReadVariableOpsequential_1/conv2d_1/kernel_1*
_class
loc:@Variable_28*&
_output_shapes
: @*
dtype0
�
Variable_28VarHandleOp*
_class
loc:@Variable_28*
_output_shapes
: *

debug_nameVariable_28/*
dtype0*
shape: @*
shared_nameVariable_28
g
,Variable_28/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_28*
_output_shapes
: 
h
Variable_28/AssignAssignVariableOpVariable_28&Variable_28/Initializer/ReadVariableOp*
dtype0
s
Variable_28/Read/ReadVariableOpReadVariableOpVariable_28*&
_output_shapes
: @*
dtype0
�
2sequential_1/batch_normalization/moving_variance_1VarHandleOp*
_output_shapes
: *C

debug_name53sequential_1/batch_normalization/moving_variance_1/*
dtype0*
shape: *C
shared_name42sequential_1/batch_normalization/moving_variance_1
�
Fsequential_1/batch_normalization/moving_variance_1/Read/ReadVariableOpReadVariableOp2sequential_1/batch_normalization/moving_variance_1*
_output_shapes
: *
dtype0
�
&Variable_29/Initializer/ReadVariableOpReadVariableOp2sequential_1/batch_normalization/moving_variance_1*
_class
loc:@Variable_29*
_output_shapes
: *
dtype0
�
Variable_29VarHandleOp*
_class
loc:@Variable_29*
_output_shapes
: *

debug_nameVariable_29/*
dtype0*
shape: *
shared_nameVariable_29
g
,Variable_29/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_29*
_output_shapes
: 
h
Variable_29/AssignAssignVariableOpVariable_29&Variable_29/Initializer/ReadVariableOp*
dtype0
g
Variable_29/Read/ReadVariableOpReadVariableOpVariable_29*
_output_shapes
: *
dtype0
�
.sequential_1/batch_normalization/moving_mean_1VarHandleOp*
_output_shapes
: *?

debug_name1/sequential_1/batch_normalization/moving_mean_1/*
dtype0*
shape: *?
shared_name0.sequential_1/batch_normalization/moving_mean_1
�
Bsequential_1/batch_normalization/moving_mean_1/Read/ReadVariableOpReadVariableOp.sequential_1/batch_normalization/moving_mean_1*
_output_shapes
: *
dtype0
�
&Variable_30/Initializer/ReadVariableOpReadVariableOp.sequential_1/batch_normalization/moving_mean_1*
_class
loc:@Variable_30*
_output_shapes
: *
dtype0
�
Variable_30VarHandleOp*
_class
loc:@Variable_30*
_output_shapes
: *

debug_nameVariable_30/*
dtype0*
shape: *
shared_nameVariable_30
g
,Variable_30/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_30*
_output_shapes
: 
h
Variable_30/AssignAssignVariableOpVariable_30&Variable_30/Initializer/ReadVariableOp*
dtype0
g
Variable_30/Read/ReadVariableOpReadVariableOpVariable_30*
_output_shapes
: *
dtype0
�
'sequential_1/batch_normalization/beta_1VarHandleOp*
_output_shapes
: *8

debug_name*(sequential_1/batch_normalization/beta_1/*
dtype0*
shape: *8
shared_name)'sequential_1/batch_normalization/beta_1
�
;sequential_1/batch_normalization/beta_1/Read/ReadVariableOpReadVariableOp'sequential_1/batch_normalization/beta_1*
_output_shapes
: *
dtype0
�
&Variable_31/Initializer/ReadVariableOpReadVariableOp'sequential_1/batch_normalization/beta_1*
_class
loc:@Variable_31*
_output_shapes
: *
dtype0
�
Variable_31VarHandleOp*
_class
loc:@Variable_31*
_output_shapes
: *

debug_nameVariable_31/*
dtype0*
shape: *
shared_nameVariable_31
g
,Variable_31/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_31*
_output_shapes
: 
h
Variable_31/AssignAssignVariableOpVariable_31&Variable_31/Initializer/ReadVariableOp*
dtype0
g
Variable_31/Read/ReadVariableOpReadVariableOpVariable_31*
_output_shapes
: *
dtype0
�
(sequential_1/batch_normalization/gamma_1VarHandleOp*
_output_shapes
: *9

debug_name+)sequential_1/batch_normalization/gamma_1/*
dtype0*
shape: *9
shared_name*(sequential_1/batch_normalization/gamma_1
�
<sequential_1/batch_normalization/gamma_1/Read/ReadVariableOpReadVariableOp(sequential_1/batch_normalization/gamma_1*
_output_shapes
: *
dtype0
�
&Variable_32/Initializer/ReadVariableOpReadVariableOp(sequential_1/batch_normalization/gamma_1*
_class
loc:@Variable_32*
_output_shapes
: *
dtype0
�
Variable_32VarHandleOp*
_class
loc:@Variable_32*
_output_shapes
: *

debug_nameVariable_32/*
dtype0*
shape: *
shared_nameVariable_32
g
,Variable_32/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_32*
_output_shapes
: 
h
Variable_32/AssignAssignVariableOpVariable_32&Variable_32/Initializer/ReadVariableOp*
dtype0
g
Variable_32/Read/ReadVariableOpReadVariableOpVariable_32*
_output_shapes
: *
dtype0
�
sequential_1/conv2d/bias_1VarHandleOp*
_output_shapes
: *+

debug_namesequential_1/conv2d/bias_1/*
dtype0*
shape: *+
shared_namesequential_1/conv2d/bias_1
�
.sequential_1/conv2d/bias_1/Read/ReadVariableOpReadVariableOpsequential_1/conv2d/bias_1*
_output_shapes
: *
dtype0
�
&Variable_33/Initializer/ReadVariableOpReadVariableOpsequential_1/conv2d/bias_1*
_class
loc:@Variable_33*
_output_shapes
: *
dtype0
�
Variable_33VarHandleOp*
_class
loc:@Variable_33*
_output_shapes
: *

debug_nameVariable_33/*
dtype0*
shape: *
shared_nameVariable_33
g
,Variable_33/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_33*
_output_shapes
: 
h
Variable_33/AssignAssignVariableOpVariable_33&Variable_33/Initializer/ReadVariableOp*
dtype0
g
Variable_33/Read/ReadVariableOpReadVariableOpVariable_33*
_output_shapes
: *
dtype0
�
sequential_1/conv2d/kernel_1VarHandleOp*
_output_shapes
: *-

debug_namesequential_1/conv2d/kernel_1/*
dtype0*
shape: *-
shared_namesequential_1/conv2d/kernel_1
�
0sequential_1/conv2d/kernel_1/Read/ReadVariableOpReadVariableOpsequential_1/conv2d/kernel_1*&
_output_shapes
: *
dtype0
�
&Variable_34/Initializer/ReadVariableOpReadVariableOpsequential_1/conv2d/kernel_1*
_class
loc:@Variable_34*&
_output_shapes
: *
dtype0
�
Variable_34VarHandleOp*
_class
loc:@Variable_34*
_output_shapes
: *

debug_nameVariable_34/*
dtype0*
shape: *
shared_nameVariable_34
g
,Variable_34/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_34*
_output_shapes
: 
h
Variable_34/AssignAssignVariableOpVariable_34&Variable_34/Initializer/ReadVariableOp*
dtype0
s
Variable_34/Read/ReadVariableOpReadVariableOpVariable_34*&
_output_shapes
: *
dtype0
�
serve_keras_tensor_4Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserve_keras_tensor_4sequential_1/conv2d/kernel_1sequential_1/conv2d/bias_1.sequential_1/batch_normalization/moving_mean_12sequential_1/batch_normalization/moving_variance_1(sequential_1/batch_normalization/gamma_1'sequential_1/batch_normalization/beta_1sequential_1/conv2d_1/kernel_1sequential_1/conv2d_1/bias_10sequential_1/batch_normalization_1/moving_mean_14sequential_1/batch_normalization_1/moving_variance_1*sequential_1/batch_normalization_1/gamma_1)sequential_1/batch_normalization_1/beta_1sequential_1/conv2d_2/kernel_1sequential_1/conv2d_2/bias_10sequential_1/batch_normalization_2/moving_mean_14sequential_1/batch_normalization_2/moving_variance_1*sequential_1/batch_normalization_2/gamma_1)sequential_1/batch_normalization_2/beta_1sequential_1/conv2d_3/kernel_1sequential_1/conv2d_3/bias_10sequential_1/batch_normalization_3/moving_mean_14sequential_1/batch_normalization_3/moving_variance_1*sequential_1/batch_normalization_3/gamma_1)sequential_1/batch_normalization_3/beta_1sequential_1/conv2d_4/kernel_1sequential_1/conv2d_4/bias_10sequential_1/batch_normalization_4/moving_mean_14sequential_1/batch_normalization_4/moving_variance_1*sequential_1/batch_normalization_4/gamma_1)sequential_1/batch_normalization_4/beta_1sequential_1/dense/kernel_1sequential_1/dense/bias_1sequential_1/dense_1/kernel_1sequential_1/dense_1/bias_1*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*D
_read_only_resource_inputs&
$"	
 !"*5
config_proto%#

CPU

GPU2*0J 8� �J *5
f0R.
,__inference_signature_wrapper___call___85212
�
serving_default_keras_tensor_4Placeholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_keras_tensor_4sequential_1/conv2d/kernel_1sequential_1/conv2d/bias_1.sequential_1/batch_normalization/moving_mean_12sequential_1/batch_normalization/moving_variance_1(sequential_1/batch_normalization/gamma_1'sequential_1/batch_normalization/beta_1sequential_1/conv2d_1/kernel_1sequential_1/conv2d_1/bias_10sequential_1/batch_normalization_1/moving_mean_14sequential_1/batch_normalization_1/moving_variance_1*sequential_1/batch_normalization_1/gamma_1)sequential_1/batch_normalization_1/beta_1sequential_1/conv2d_2/kernel_1sequential_1/conv2d_2/bias_10sequential_1/batch_normalization_2/moving_mean_14sequential_1/batch_normalization_2/moving_variance_1*sequential_1/batch_normalization_2/gamma_1)sequential_1/batch_normalization_2/beta_1sequential_1/conv2d_3/kernel_1sequential_1/conv2d_3/bias_10sequential_1/batch_normalization_3/moving_mean_14sequential_1/batch_normalization_3/moving_variance_1*sequential_1/batch_normalization_3/gamma_1)sequential_1/batch_normalization_3/beta_1sequential_1/conv2d_4/kernel_1sequential_1/conv2d_4/bias_10sequential_1/batch_normalization_4/moving_mean_14sequential_1/batch_normalization_4/moving_variance_1*sequential_1/batch_normalization_4/gamma_1)sequential_1/batch_normalization_4/beta_1sequential_1/dense/kernel_1sequential_1/dense/bias_1sequential_1/dense_1/kernel_1sequential_1/dense_1/bias_1*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*D
_read_only_resource_inputs&
$"	
 !"*5
config_proto%#

CPU

GPU2*0J 8� �J *5
f0R.
,__inference_signature_wrapper___call___85285

NoOpNoOp
�;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�:
value�:B�: B�:
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30
'31
(32
)33
*34*
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
!17
"18
#19
&20
'21
)22
*23*
R
0
1
2
3
4
5
6
7
$8
%9
(10*
�
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31
K32
L33*
* 

Mtrace_0* 
"
	Nserve
Oserving_default* 
KE
VARIABLE_VALUEVariable_34&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_33&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_32&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_31&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_30&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_29&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_28&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_27&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_26&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_25&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_24'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_23'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_22'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_21'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_20'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_19'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_18'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_17'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_16'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_15'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_14'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_13'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_12'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_11'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_10'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_9'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_8'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_7'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_6'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_5'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_4'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_3'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_2'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_1'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEVariable'variables/34/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE'sequential_1/batch_normalization/beta_1+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEsequential_1/conv2d/kernel_1+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEsequential_1/conv2d_1/kernel_1+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEsequential_1/conv2d_2/bias_1+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE)sequential_1/batch_normalization_2/beta_1+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE*sequential_1/batch_normalization_3/gamma_1+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE)sequential_1/batch_normalization_4/beta_1+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE*sequential_1/batch_normalization_1/gamma_1+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEsequential_1/dense_1/bias_1+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEsequential_1/conv2d_3/kernel_1+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEsequential_1/conv2d_4/kernel_1,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE*sequential_1/batch_normalization_4/gamma_1,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEsequential_1/dense_1/kernel_1,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEsequential_1/conv2d_1/bias_1,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEsequential_1/conv2d_2/kernel_1,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsequential_1/dense/bias_1,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE(sequential_1/batch_normalization/gamma_1,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE)sequential_1/batch_normalization_3/beta_1,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEsequential_1/conv2d/bias_1,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE)sequential_1/batch_normalization_1/beta_1,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE*sequential_1/batch_normalization_2/gamma_1,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEsequential_1/conv2d_3/bias_1,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEsequential_1/conv2d_4/bias_1,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEsequential_1/dense/kernel_1,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE0sequential_1/batch_normalization_3/moving_mean_1,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE4sequential_1/batch_normalization_4/moving_variance_1,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE0sequential_1/batch_normalization_1/moving_mean_1,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUE.sequential_1/batch_normalization/moving_mean_1,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE4sequential_1/batch_normalization_1/moving_variance_1,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE0sequential_1/batch_normalization_2/moving_mean_1,_all_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE0sequential_1/batch_normalization_4/moving_mean_1,_all_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE2sequential_1/batch_normalization/moving_variance_1,_all_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE4sequential_1/batch_normalization_2/moving_variance_1,_all_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE4sequential_1/batch_normalization_3/moving_variance_1,_all_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_34Variable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable'sequential_1/batch_normalization/beta_1sequential_1/conv2d/kernel_1sequential_1/conv2d_1/kernel_1sequential_1/conv2d_2/bias_1)sequential_1/batch_normalization_2/beta_1*sequential_1/batch_normalization_3/gamma_1)sequential_1/batch_normalization_4/beta_1*sequential_1/batch_normalization_1/gamma_1sequential_1/dense_1/bias_1sequential_1/conv2d_3/kernel_1sequential_1/conv2d_4/kernel_1*sequential_1/batch_normalization_4/gamma_1sequential_1/dense_1/kernel_1sequential_1/conv2d_1/bias_1sequential_1/conv2d_2/kernel_1sequential_1/dense/bias_1(sequential_1/batch_normalization/gamma_1)sequential_1/batch_normalization_3/beta_1sequential_1/conv2d/bias_1)sequential_1/batch_normalization_1/beta_1*sequential_1/batch_normalization_2/gamma_1sequential_1/conv2d_3/bias_1sequential_1/conv2d_4/bias_1sequential_1/dense/kernel_10sequential_1/batch_normalization_3/moving_mean_14sequential_1/batch_normalization_4/moving_variance_10sequential_1/batch_normalization_1/moving_mean_1.sequential_1/batch_normalization/moving_mean_14sequential_1/batch_normalization_1/moving_variance_10sequential_1/batch_normalization_2/moving_mean_10sequential_1/batch_normalization_4/moving_mean_12sequential_1/batch_normalization/moving_variance_14sequential_1/batch_normalization_2/moving_variance_14sequential_1/batch_normalization_3/moving_variance_1Const*R
TinK
I2G*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8� �J *'
f"R 
__inference__traced_save_85863
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable_34Variable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variable'sequential_1/batch_normalization/beta_1sequential_1/conv2d/kernel_1sequential_1/conv2d_1/kernel_1sequential_1/conv2d_2/bias_1)sequential_1/batch_normalization_2/beta_1*sequential_1/batch_normalization_3/gamma_1)sequential_1/batch_normalization_4/beta_1*sequential_1/batch_normalization_1/gamma_1sequential_1/dense_1/bias_1sequential_1/conv2d_3/kernel_1sequential_1/conv2d_4/kernel_1*sequential_1/batch_normalization_4/gamma_1sequential_1/dense_1/kernel_1sequential_1/conv2d_1/bias_1sequential_1/conv2d_2/kernel_1sequential_1/dense/bias_1(sequential_1/batch_normalization/gamma_1)sequential_1/batch_normalization_3/beta_1sequential_1/conv2d/bias_1)sequential_1/batch_normalization_1/beta_1*sequential_1/batch_normalization_2/gamma_1sequential_1/conv2d_3/bias_1sequential_1/conv2d_4/bias_1sequential_1/dense/kernel_10sequential_1/batch_normalization_3/moving_mean_14sequential_1/batch_normalization_4/moving_variance_10sequential_1/batch_normalization_1/moving_mean_1.sequential_1/batch_normalization/moving_mean_14sequential_1/batch_normalization_1/moving_variance_10sequential_1/batch_normalization_2/moving_mean_10sequential_1/batch_normalization_4/moving_mean_12sequential_1/batch_normalization/moving_variance_14sequential_1/batch_normalization_2/moving_variance_14sequential_1/batch_normalization_3/moving_variance_1*Q
TinJ
H2F*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *5
config_proto%#

CPU

GPU2*0J 8� �J **
f%R#
!__inference__traced_restore_86079��
�
�.
!__inference__traced_restore_86079
file_prefix6
assignvariableop_variable_34: ,
assignvariableop_1_variable_33: ,
assignvariableop_2_variable_32: ,
assignvariableop_3_variable_31: ,
assignvariableop_4_variable_30: ,
assignvariableop_5_variable_29: 8
assignvariableop_6_variable_28: @,
assignvariableop_7_variable_27:@,
assignvariableop_8_variable_26:@,
assignvariableop_9_variable_25:@-
assignvariableop_10_variable_24:@-
assignvariableop_11_variable_23:@:
assignvariableop_12_variable_22:@�.
assignvariableop_13_variable_21:	�.
assignvariableop_14_variable_20:	�.
assignvariableop_15_variable_19:	�.
assignvariableop_16_variable_18:	�.
assignvariableop_17_variable_17:	�;
assignvariableop_18_variable_16:��.
assignvariableop_19_variable_15:	�.
assignvariableop_20_variable_14:	�.
assignvariableop_21_variable_13:	�.
assignvariableop_22_variable_12:	�.
assignvariableop_23_variable_11:	�;
assignvariableop_24_variable_10:��-
assignvariableop_25_variable_9:	�-
assignvariableop_26_variable_8:	�-
assignvariableop_27_variable_7:	�-
assignvariableop_28_variable_6:	�-
assignvariableop_29_variable_5:	�2
assignvariableop_30_variable_4:
�b�-
assignvariableop_31_variable_3:	�,
assignvariableop_32_variable_2:	1
assignvariableop_33_variable_1:	�*
assignvariableop_34_variable:I
;assignvariableop_35_sequential_1_batch_normalization_beta_1: J
0assignvariableop_36_sequential_1_conv2d_kernel_1: L
2assignvariableop_37_sequential_1_conv2d_1_kernel_1: @?
0assignvariableop_38_sequential_1_conv2d_2_bias_1:	�L
=assignvariableop_39_sequential_1_batch_normalization_2_beta_1:	�M
>assignvariableop_40_sequential_1_batch_normalization_3_gamma_1:	�L
=assignvariableop_41_sequential_1_batch_normalization_4_beta_1:	�L
>assignvariableop_42_sequential_1_batch_normalization_1_gamma_1:@=
/assignvariableop_43_sequential_1_dense_1_bias_1:N
2assignvariableop_44_sequential_1_conv2d_3_kernel_1:��N
2assignvariableop_45_sequential_1_conv2d_4_kernel_1:��M
>assignvariableop_46_sequential_1_batch_normalization_4_gamma_1:	�D
1assignvariableop_47_sequential_1_dense_1_kernel_1:	�>
0assignvariableop_48_sequential_1_conv2d_1_bias_1:@M
2assignvariableop_49_sequential_1_conv2d_2_kernel_1:@�<
-assignvariableop_50_sequential_1_dense_bias_1:	�J
<assignvariableop_51_sequential_1_batch_normalization_gamma_1: L
=assignvariableop_52_sequential_1_batch_normalization_3_beta_1:	�<
.assignvariableop_53_sequential_1_conv2d_bias_1: K
=assignvariableop_54_sequential_1_batch_normalization_1_beta_1:@M
>assignvariableop_55_sequential_1_batch_normalization_2_gamma_1:	�?
0assignvariableop_56_sequential_1_conv2d_3_bias_1:	�?
0assignvariableop_57_sequential_1_conv2d_4_bias_1:	�C
/assignvariableop_58_sequential_1_dense_kernel_1:
�b�S
Dassignvariableop_59_sequential_1_batch_normalization_3_moving_mean_1:	�W
Hassignvariableop_60_sequential_1_batch_normalization_4_moving_variance_1:	�R
Dassignvariableop_61_sequential_1_batch_normalization_1_moving_mean_1:@P
Bassignvariableop_62_sequential_1_batch_normalization_moving_mean_1: V
Hassignvariableop_63_sequential_1_batch_normalization_1_moving_variance_1:@S
Dassignvariableop_64_sequential_1_batch_normalization_2_moving_mean_1:	�S
Dassignvariableop_65_sequential_1_batch_normalization_4_moving_mean_1:	�T
Fassignvariableop_66_sequential_1_batch_normalization_moving_variance_1: W
Hassignvariableop_67_sequential_1_batch_normalization_2_moving_variance_1:	�W
Hassignvariableop_68_sequential_1_batch_normalization_3_moving_variance_1:	�
identity_70��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�
value�B�FB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/29/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/30/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/31/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/32/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/33/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�
value�B�FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*T
dtypesJ
H2F	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_34Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_33Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_32Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_31Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_30Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_29Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_28Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_27Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_26Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_25Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_24Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_23Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_22Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_21Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_20Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_19Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_18Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_17Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_16Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_15Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_14Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_13Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_12Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_11Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_10Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variable_9Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_variable_8Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_variable_7Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_variable_6Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_variable_5Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_variable_4Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_variable_3Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_variable_2Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_variable_1Identity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_variableIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp;assignvariableop_35_sequential_1_batch_normalization_beta_1Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp0assignvariableop_36_sequential_1_conv2d_kernel_1Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp2assignvariableop_37_sequential_1_conv2d_1_kernel_1Identity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp0assignvariableop_38_sequential_1_conv2d_2_bias_1Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp=assignvariableop_39_sequential_1_batch_normalization_2_beta_1Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp>assignvariableop_40_sequential_1_batch_normalization_3_gamma_1Identity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp=assignvariableop_41_sequential_1_batch_normalization_4_beta_1Identity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp>assignvariableop_42_sequential_1_batch_normalization_1_gamma_1Identity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp/assignvariableop_43_sequential_1_dense_1_bias_1Identity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp2assignvariableop_44_sequential_1_conv2d_3_kernel_1Identity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp2assignvariableop_45_sequential_1_conv2d_4_kernel_1Identity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp>assignvariableop_46_sequential_1_batch_normalization_4_gamma_1Identity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp1assignvariableop_47_sequential_1_dense_1_kernel_1Identity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp0assignvariableop_48_sequential_1_conv2d_1_bias_1Identity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp2assignvariableop_49_sequential_1_conv2d_2_kernel_1Identity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp-assignvariableop_50_sequential_1_dense_bias_1Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp<assignvariableop_51_sequential_1_batch_normalization_gamma_1Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp=assignvariableop_52_sequential_1_batch_normalization_3_beta_1Identity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp.assignvariableop_53_sequential_1_conv2d_bias_1Identity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp=assignvariableop_54_sequential_1_batch_normalization_1_beta_1Identity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp>assignvariableop_55_sequential_1_batch_normalization_2_gamma_1Identity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp0assignvariableop_56_sequential_1_conv2d_3_bias_1Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp0assignvariableop_57_sequential_1_conv2d_4_bias_1Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp/assignvariableop_58_sequential_1_dense_kernel_1Identity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpDassignvariableop_59_sequential_1_batch_normalization_3_moving_mean_1Identity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpHassignvariableop_60_sequential_1_batch_normalization_4_moving_variance_1Identity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpDassignvariableop_61_sequential_1_batch_normalization_1_moving_mean_1Identity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpBassignvariableop_62_sequential_1_batch_normalization_moving_mean_1Identity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpHassignvariableop_63_sequential_1_batch_normalization_1_moving_variance_1Identity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpDassignvariableop_64_sequential_1_batch_normalization_2_moving_mean_1Identity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOpDassignvariableop_65_sequential_1_batch_normalization_4_moving_mean_1Identity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOpFassignvariableop_66_sequential_1_batch_normalization_moving_variance_1Identity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOpHassignvariableop_67_sequential_1_batch_normalization_2_moving_variance_1Identity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOpHassignvariableop_68_sequential_1_batch_normalization_3_moving_variance_1Identity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_69Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_70IdentityIdentity_69:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_70Identity_70:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:TEP
N
_user_specified_name64sequential_1/batch_normalization_3/moving_variance_1:TDP
N
_user_specified_name64sequential_1/batch_normalization_2/moving_variance_1:RCN
L
_user_specified_name42sequential_1/batch_normalization/moving_variance_1:PBL
J
_user_specified_name20sequential_1/batch_normalization_4/moving_mean_1:PAL
J
_user_specified_name20sequential_1/batch_normalization_2/moving_mean_1:T@P
N
_user_specified_name64sequential_1/batch_normalization_1/moving_variance_1:N?J
H
_user_specified_name0.sequential_1/batch_normalization/moving_mean_1:P>L
J
_user_specified_name20sequential_1/batch_normalization_1/moving_mean_1:T=P
N
_user_specified_name64sequential_1/batch_normalization_4/moving_variance_1:P<L
J
_user_specified_name20sequential_1/batch_normalization_3/moving_mean_1:;;7
5
_user_specified_namesequential_1/dense/kernel_1:<:8
6
_user_specified_namesequential_1/conv2d_4/bias_1:<98
6
_user_specified_namesequential_1/conv2d_3/bias_1:J8F
D
_user_specified_name,*sequential_1/batch_normalization_2/gamma_1:I7E
C
_user_specified_name+)sequential_1/batch_normalization_1/beta_1::66
4
_user_specified_namesequential_1/conv2d/bias_1:I5E
C
_user_specified_name+)sequential_1/batch_normalization_3/beta_1:H4D
B
_user_specified_name*(sequential_1/batch_normalization/gamma_1:935
3
_user_specified_namesequential_1/dense/bias_1:>2:
8
_user_specified_name sequential_1/conv2d_2/kernel_1:<18
6
_user_specified_namesequential_1/conv2d_1/bias_1:=09
7
_user_specified_namesequential_1/dense_1/kernel_1:J/F
D
_user_specified_name,*sequential_1/batch_normalization_4/gamma_1:>.:
8
_user_specified_name sequential_1/conv2d_4/kernel_1:>-:
8
_user_specified_name sequential_1/conv2d_3/kernel_1:;,7
5
_user_specified_namesequential_1/dense_1/bias_1:J+F
D
_user_specified_name,*sequential_1/batch_normalization_1/gamma_1:I*E
C
_user_specified_name+)sequential_1/batch_normalization_4/beta_1:J)F
D
_user_specified_name,*sequential_1/batch_normalization_3/gamma_1:I(E
C
_user_specified_name+)sequential_1/batch_normalization_2/beta_1:<'8
6
_user_specified_namesequential_1/conv2d_2/bias_1:>&:
8
_user_specified_name sequential_1/conv2d_1/kernel_1:<%8
6
_user_specified_namesequential_1/conv2d/kernel_1:G$C
A
_user_specified_name)'sequential_1/batch_normalization/beta_1:(#$
"
_user_specified_name
Variable:*"&
$
_user_specified_name
Variable_1:*!&
$
_user_specified_name
Variable_2:* &
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_24:+
'
%
_user_specified_nameVariable_25:+	'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_27:+'
%
_user_specified_nameVariable_28:+'
%
_user_specified_nameVariable_29:+'
%
_user_specified_nameVariable_30:+'
%
_user_specified_nameVariable_31:+'
%
_user_specified_nameVariable_32:+'
%
_user_specified_nameVariable_33:+'
%
_user_specified_nameVariable_34:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
,__inference_signature_wrapper___call___85285
keras_tensor_4!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�&

unknown_17:��

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�&

unknown_23:��

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:
�b�

unknown_30:	�

unknown_31:	�

unknown_32:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallkeras_tensor_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*D
_read_only_resource_inputs&
$"	
 !"*5
config_proto%#

CPU

GPU2*0J 8� �J *#
fR
__inference___call___85138o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%"!

_user_specified_name85281:%!!

_user_specified_name85279:% !

_user_specified_name85277:%!

_user_specified_name85275:%!

_user_specified_name85273:%!

_user_specified_name85271:%!

_user_specified_name85269:%!

_user_specified_name85267:%!

_user_specified_name85265:%!

_user_specified_name85263:%!

_user_specified_name85261:%!

_user_specified_name85259:%!

_user_specified_name85257:%!

_user_specified_name85255:%!

_user_specified_name85253:%!

_user_specified_name85251:%!

_user_specified_name85249:%!

_user_specified_name85247:%!

_user_specified_name85245:%!

_user_specified_name85243:%!

_user_specified_name85241:%!

_user_specified_name85239:%!

_user_specified_name85237:%!

_user_specified_name85235:%
!

_user_specified_name85233:%	!

_user_specified_name85231:%!

_user_specified_name85229:%!

_user_specified_name85227:%!

_user_specified_name85225:%!

_user_specified_name85223:%!

_user_specified_name85221:%!

_user_specified_name85219:%!

_user_specified_name85217:%!

_user_specified_name85215:a ]
1
_output_shapes
:�����������
(
_user_specified_namekeras_tensor_4
�
�
,__inference_signature_wrapper___call___85212
keras_tensor_4!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@%

unknown_11:@�

unknown_12:	�

unknown_13:	�

unknown_14:	�

unknown_15:	�

unknown_16:	�&

unknown_17:��

unknown_18:	�

unknown_19:	�

unknown_20:	�

unknown_21:	�

unknown_22:	�&

unknown_23:��

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:
�b�

unknown_30:	�

unknown_31:	�

unknown_32:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallkeras_tensor_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*D
_read_only_resource_inputs&
$"	
 !"*5
config_proto%#

CPU

GPU2*0J 8� �J *#
fR
__inference___call___85138o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:%"!

_user_specified_name85208:%!!

_user_specified_name85206:% !

_user_specified_name85204:%!

_user_specified_name85202:%!

_user_specified_name85200:%!

_user_specified_name85198:%!

_user_specified_name85196:%!

_user_specified_name85194:%!

_user_specified_name85192:%!

_user_specified_name85190:%!

_user_specified_name85188:%!

_user_specified_name85186:%!

_user_specified_name85184:%!

_user_specified_name85182:%!

_user_specified_name85180:%!

_user_specified_name85178:%!

_user_specified_name85176:%!

_user_specified_name85174:%!

_user_specified_name85172:%!

_user_specified_name85170:%!

_user_specified_name85168:%!

_user_specified_name85166:%!

_user_specified_name85164:%!

_user_specified_name85162:%
!

_user_specified_name85160:%	!

_user_specified_name85158:%!

_user_specified_name85156:%!

_user_specified_name85154:%!

_user_specified_name85152:%!

_user_specified_name85150:%!

_user_specified_name85148:%!

_user_specified_name85146:%!

_user_specified_name85144:%!

_user_specified_name85142:a ]
1
_output_shapes
:�����������
(
_user_specified_namekeras_tensor_4
��
�B
__inference__traced_save_85863
file_prefix<
"read_disablecopyonread_variable_34: 2
$read_1_disablecopyonread_variable_33: 2
$read_2_disablecopyonread_variable_32: 2
$read_3_disablecopyonread_variable_31: 2
$read_4_disablecopyonread_variable_30: 2
$read_5_disablecopyonread_variable_29: >
$read_6_disablecopyonread_variable_28: @2
$read_7_disablecopyonread_variable_27:@2
$read_8_disablecopyonread_variable_26:@2
$read_9_disablecopyonread_variable_25:@3
%read_10_disablecopyonread_variable_24:@3
%read_11_disablecopyonread_variable_23:@@
%read_12_disablecopyonread_variable_22:@�4
%read_13_disablecopyonread_variable_21:	�4
%read_14_disablecopyonread_variable_20:	�4
%read_15_disablecopyonread_variable_19:	�4
%read_16_disablecopyonread_variable_18:	�4
%read_17_disablecopyonread_variable_17:	�A
%read_18_disablecopyonread_variable_16:��4
%read_19_disablecopyonread_variable_15:	�4
%read_20_disablecopyonread_variable_14:	�4
%read_21_disablecopyonread_variable_13:	�4
%read_22_disablecopyonread_variable_12:	�4
%read_23_disablecopyonread_variable_11:	�A
%read_24_disablecopyonread_variable_10:��3
$read_25_disablecopyonread_variable_9:	�3
$read_26_disablecopyonread_variable_8:	�3
$read_27_disablecopyonread_variable_7:	�3
$read_28_disablecopyonread_variable_6:	�3
$read_29_disablecopyonread_variable_5:	�8
$read_30_disablecopyonread_variable_4:
�b�3
$read_31_disablecopyonread_variable_3:	�2
$read_32_disablecopyonread_variable_2:	7
$read_33_disablecopyonread_variable_1:	�0
"read_34_disablecopyonread_variable:O
Aread_35_disablecopyonread_sequential_1_batch_normalization_beta_1: P
6read_36_disablecopyonread_sequential_1_conv2d_kernel_1: R
8read_37_disablecopyonread_sequential_1_conv2d_1_kernel_1: @E
6read_38_disablecopyonread_sequential_1_conv2d_2_bias_1:	�R
Cread_39_disablecopyonread_sequential_1_batch_normalization_2_beta_1:	�S
Dread_40_disablecopyonread_sequential_1_batch_normalization_3_gamma_1:	�R
Cread_41_disablecopyonread_sequential_1_batch_normalization_4_beta_1:	�R
Dread_42_disablecopyonread_sequential_1_batch_normalization_1_gamma_1:@C
5read_43_disablecopyonread_sequential_1_dense_1_bias_1:T
8read_44_disablecopyonread_sequential_1_conv2d_3_kernel_1:��T
8read_45_disablecopyonread_sequential_1_conv2d_4_kernel_1:��S
Dread_46_disablecopyonread_sequential_1_batch_normalization_4_gamma_1:	�J
7read_47_disablecopyonread_sequential_1_dense_1_kernel_1:	�D
6read_48_disablecopyonread_sequential_1_conv2d_1_bias_1:@S
8read_49_disablecopyonread_sequential_1_conv2d_2_kernel_1:@�B
3read_50_disablecopyonread_sequential_1_dense_bias_1:	�P
Bread_51_disablecopyonread_sequential_1_batch_normalization_gamma_1: R
Cread_52_disablecopyonread_sequential_1_batch_normalization_3_beta_1:	�B
4read_53_disablecopyonread_sequential_1_conv2d_bias_1: Q
Cread_54_disablecopyonread_sequential_1_batch_normalization_1_beta_1:@S
Dread_55_disablecopyonread_sequential_1_batch_normalization_2_gamma_1:	�E
6read_56_disablecopyonread_sequential_1_conv2d_3_bias_1:	�E
6read_57_disablecopyonread_sequential_1_conv2d_4_bias_1:	�I
5read_58_disablecopyonread_sequential_1_dense_kernel_1:
�b�Y
Jread_59_disablecopyonread_sequential_1_batch_normalization_3_moving_mean_1:	�]
Nread_60_disablecopyonread_sequential_1_batch_normalization_4_moving_variance_1:	�X
Jread_61_disablecopyonread_sequential_1_batch_normalization_1_moving_mean_1:@V
Hread_62_disablecopyonread_sequential_1_batch_normalization_moving_mean_1: \
Nread_63_disablecopyonread_sequential_1_batch_normalization_1_moving_variance_1:@Y
Jread_64_disablecopyonread_sequential_1_batch_normalization_2_moving_mean_1:	�Y
Jread_65_disablecopyonread_sequential_1_batch_normalization_4_moving_mean_1:	�Z
Lread_66_disablecopyonread_sequential_1_batch_normalization_moving_variance_1: ]
Nread_67_disablecopyonread_sequential_1_batch_normalization_2_moving_variance_1:	�]
Nread_68_disablecopyonread_sequential_1_batch_normalization_3_moving_variance_1:	�
savev2_const
identity_139��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_34*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_34^Read/DisableCopyOnRead*&
_output_shapes
: *
dtype0b
IdentityIdentityRead/ReadVariableOp:value:0*
T0*&
_output_shapes
: i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
: i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_33*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_33^Read_1/DisableCopyOnRead*
_output_shapes
: *
dtype0Z

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
: _

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_32*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_32^Read_2/DisableCopyOnRead*
_output_shapes
: *
dtype0Z

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes
: _

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_31*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_31^Read_3/DisableCopyOnRead*
_output_shapes
: *
dtype0Z

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
: _

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_30*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_30^Read_4/DisableCopyOnRead*
_output_shapes
: *
dtype0Z

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*
_output_shapes
: _

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_29*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_29^Read_5/DisableCopyOnRead*
_output_shapes
: *
dtype0[
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_28*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_28^Read_6/DisableCopyOnRead*&
_output_shapes
: @*
dtype0g
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*&
_output_shapes
: @m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
: @i
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_variable_27*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_variable_27^Read_7/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@i
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_variable_26*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_variable_26^Read_8/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@i
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_variable_25*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_variable_25^Read_9/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_variable_24*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_variable_24^Read_10/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_variable_23*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_variable_23^Read_11/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_variable_22*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_variable_22^Read_12/DisableCopyOnRead*'
_output_shapes
:@�*
dtype0i
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�n
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�k
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_variable_21*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_variable_21^Read_13/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_14/DisableCopyOnReadDisableCopyOnRead%read_14_disablecopyonread_variable_20*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp%read_14_disablecopyonread_variable_20^Read_14/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_15/DisableCopyOnReadDisableCopyOnRead%read_15_disablecopyonread_variable_19*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp%read_15_disablecopyonread_variable_19^Read_15/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_16/DisableCopyOnReadDisableCopyOnRead%read_16_disablecopyonread_variable_18*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp%read_16_disablecopyonread_variable_18^Read_16/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_17/DisableCopyOnReadDisableCopyOnRead%read_17_disablecopyonread_variable_17*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp%read_17_disablecopyonread_variable_17^Read_17/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_18/DisableCopyOnReadDisableCopyOnRead%read_18_disablecopyonread_variable_16*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp%read_18_disablecopyonread_variable_16^Read_18/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*(
_output_shapes
:��k
Read_19/DisableCopyOnReadDisableCopyOnRead%read_19_disablecopyonread_variable_15*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp%read_19_disablecopyonread_variable_15^Read_19/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_20/DisableCopyOnReadDisableCopyOnRead%read_20_disablecopyonread_variable_14*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp%read_20_disablecopyonread_variable_14^Read_20/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_21/DisableCopyOnReadDisableCopyOnRead%read_21_disablecopyonread_variable_13*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp%read_21_disablecopyonread_variable_13^Read_21/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_22/DisableCopyOnReadDisableCopyOnRead%read_22_disablecopyonread_variable_12*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp%read_22_disablecopyonread_variable_12^Read_22/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_23/DisableCopyOnReadDisableCopyOnRead%read_23_disablecopyonread_variable_11*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp%read_23_disablecopyonread_variable_11^Read_23/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_24/DisableCopyOnReadDisableCopyOnRead%read_24_disablecopyonread_variable_10*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp%read_24_disablecopyonread_variable_10^Read_24/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*(
_output_shapes
:��j
Read_25/DisableCopyOnReadDisableCopyOnRead$read_25_disablecopyonread_variable_9*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp$read_25_disablecopyonread_variable_9^Read_25/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_26/DisableCopyOnReadDisableCopyOnRead$read_26_disablecopyonread_variable_8*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp$read_26_disablecopyonread_variable_8^Read_26/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_52IdentityRead_26/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_27/DisableCopyOnReadDisableCopyOnRead$read_27_disablecopyonread_variable_7*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp$read_27_disablecopyonread_variable_7^Read_27/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_54IdentityRead_27/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_28/DisableCopyOnReadDisableCopyOnRead$read_28_disablecopyonread_variable_6*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp$read_28_disablecopyonread_variable_6^Read_28/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_56IdentityRead_28/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_29/DisableCopyOnReadDisableCopyOnRead$read_29_disablecopyonread_variable_5*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp$read_29_disablecopyonread_variable_5^Read_29/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_58IdentityRead_29/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_30/DisableCopyOnReadDisableCopyOnRead$read_30_disablecopyonread_variable_4*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp$read_30_disablecopyonread_variable_4^Read_30/DisableCopyOnRead* 
_output_shapes
:
�b�*
dtype0b
Identity_60IdentityRead_30/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�b�g
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0* 
_output_shapes
:
�b�j
Read_31/DisableCopyOnReadDisableCopyOnRead$read_31_disablecopyonread_variable_3*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp$read_31_disablecopyonread_variable_3^Read_31/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_62IdentityRead_31/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_32/DisableCopyOnReadDisableCopyOnRead$read_32_disablecopyonread_variable_2*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp$read_32_disablecopyonread_variable_2^Read_32/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_64IdentityRead_32/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0	*
_output_shapes
:j
Read_33/DisableCopyOnReadDisableCopyOnRead$read_33_disablecopyonread_variable_1*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp$read_33_disablecopyonread_variable_1^Read_33/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_66IdentityRead_33/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Read_34/DisableCopyOnReadDisableCopyOnRead"read_34_disablecopyonread_variable*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp"read_34_disablecopyonread_variable^Read_34/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_68IdentityRead_34/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_35/DisableCopyOnReadDisableCopyOnReadAread_35_disablecopyonread_sequential_1_batch_normalization_beta_1*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOpAread_35_disablecopyonread_sequential_1_batch_normalization_beta_1^Read_35/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_70IdentityRead_35/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
: |
Read_36/DisableCopyOnReadDisableCopyOnRead6read_36_disablecopyonread_sequential_1_conv2d_kernel_1*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp6read_36_disablecopyonread_sequential_1_conv2d_kernel_1^Read_36/DisableCopyOnRead*&
_output_shapes
: *
dtype0h
Identity_72IdentityRead_36/ReadVariableOp:value:0*
T0*&
_output_shapes
: m
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
: ~
Read_37/DisableCopyOnReadDisableCopyOnRead8read_37_disablecopyonread_sequential_1_conv2d_1_kernel_1*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp8read_37_disablecopyonread_sequential_1_conv2d_1_kernel_1^Read_37/DisableCopyOnRead*&
_output_shapes
: @*
dtype0h
Identity_74IdentityRead_37/ReadVariableOp:value:0*
T0*&
_output_shapes
: @m
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*&
_output_shapes
: @|
Read_38/DisableCopyOnReadDisableCopyOnRead6read_38_disablecopyonread_sequential_1_conv2d_2_bias_1*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp6read_38_disablecopyonread_sequential_1_conv2d_2_bias_1^Read_38/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_76IdentityRead_38/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_39/DisableCopyOnReadDisableCopyOnReadCread_39_disablecopyonread_sequential_1_batch_normalization_2_beta_1*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOpCread_39_disablecopyonread_sequential_1_batch_normalization_2_beta_1^Read_39/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_78IdentityRead_39/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_40/DisableCopyOnReadDisableCopyOnReadDread_40_disablecopyonread_sequential_1_batch_normalization_3_gamma_1*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOpDread_40_disablecopyonread_sequential_1_batch_normalization_3_gamma_1^Read_40/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_80IdentityRead_40/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_41/DisableCopyOnReadDisableCopyOnReadCread_41_disablecopyonread_sequential_1_batch_normalization_4_beta_1*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOpCread_41_disablecopyonread_sequential_1_batch_normalization_4_beta_1^Read_41/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_82IdentityRead_41/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_42/DisableCopyOnReadDisableCopyOnReadDread_42_disablecopyonread_sequential_1_batch_normalization_1_gamma_1*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOpDread_42_disablecopyonread_sequential_1_batch_normalization_1_gamma_1^Read_42/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_84IdentityRead_42/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_43/DisableCopyOnReadDisableCopyOnRead5read_43_disablecopyonread_sequential_1_dense_1_bias_1*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp5read_43_disablecopyonread_sequential_1_dense_1_bias_1^Read_43/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_86IdentityRead_43/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_44/DisableCopyOnReadDisableCopyOnRead8read_44_disablecopyonread_sequential_1_conv2d_3_kernel_1*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp8read_44_disablecopyonread_sequential_1_conv2d_3_kernel_1^Read_44/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_88IdentityRead_44/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*(
_output_shapes
:��~
Read_45/DisableCopyOnReadDisableCopyOnRead8read_45_disablecopyonread_sequential_1_conv2d_4_kernel_1*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp8read_45_disablecopyonread_sequential_1_conv2d_4_kernel_1^Read_45/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_90IdentityRead_45/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_46/DisableCopyOnReadDisableCopyOnReadDread_46_disablecopyonread_sequential_1_batch_normalization_4_gamma_1*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOpDread_46_disablecopyonread_sequential_1_batch_normalization_4_gamma_1^Read_46/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_92IdentityRead_46/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_47/DisableCopyOnReadDisableCopyOnRead7read_47_disablecopyonread_sequential_1_dense_1_kernel_1*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp7read_47_disablecopyonread_sequential_1_dense_1_kernel_1^Read_47/DisableCopyOnRead*
_output_shapes
:	�*
dtype0a
Identity_94IdentityRead_47/ReadVariableOp:value:0*
T0*
_output_shapes
:	�f
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:	�|
Read_48/DisableCopyOnReadDisableCopyOnRead6read_48_disablecopyonread_sequential_1_conv2d_1_bias_1*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp6read_48_disablecopyonread_sequential_1_conv2d_1_bias_1^Read_48/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_96IdentityRead_48/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_49/DisableCopyOnReadDisableCopyOnRead8read_49_disablecopyonread_sequential_1_conv2d_2_kernel_1*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp8read_49_disablecopyonread_sequential_1_conv2d_2_kernel_1^Read_49/DisableCopyOnRead*'
_output_shapes
:@�*
dtype0i
Identity_98IdentityRead_49/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�n
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�y
Read_50/DisableCopyOnReadDisableCopyOnRead3read_50_disablecopyonread_sequential_1_dense_bias_1*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp3read_50_disablecopyonread_sequential_1_dense_bias_1^Read_50/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_100IdentityRead_50/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_51/DisableCopyOnReadDisableCopyOnReadBread_51_disablecopyonread_sequential_1_batch_normalization_gamma_1*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOpBread_51_disablecopyonread_sequential_1_batch_normalization_gamma_1^Read_51/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_102IdentityRead_51/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_52/DisableCopyOnReadDisableCopyOnReadCread_52_disablecopyonread_sequential_1_batch_normalization_3_beta_1*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOpCread_52_disablecopyonread_sequential_1_batch_normalization_3_beta_1^Read_52/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_104IdentityRead_52/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:�z
Read_53/DisableCopyOnReadDisableCopyOnRead4read_53_disablecopyonread_sequential_1_conv2d_bias_1*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp4read_53_disablecopyonread_sequential_1_conv2d_bias_1^Read_53/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_106IdentityRead_53/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_54/DisableCopyOnReadDisableCopyOnReadCread_54_disablecopyonread_sequential_1_batch_normalization_1_beta_1*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOpCread_54_disablecopyonread_sequential_1_batch_normalization_1_beta_1^Read_54/DisableCopyOnRead*
_output_shapes
:@*
dtype0]
Identity_108IdentityRead_54/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_55/DisableCopyOnReadDisableCopyOnReadDread_55_disablecopyonread_sequential_1_batch_normalization_2_gamma_1*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOpDread_55_disablecopyonread_sequential_1_batch_normalization_2_gamma_1^Read_55/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_110IdentityRead_55/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_56/DisableCopyOnReadDisableCopyOnRead6read_56_disablecopyonread_sequential_1_conv2d_3_bias_1*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp6read_56_disablecopyonread_sequential_1_conv2d_3_bias_1^Read_56/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_112IdentityRead_56/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_57/DisableCopyOnReadDisableCopyOnRead6read_57_disablecopyonread_sequential_1_conv2d_4_bias_1*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp6read_57_disablecopyonread_sequential_1_conv2d_4_bias_1^Read_57/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_114IdentityRead_57/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:�{
Read_58/DisableCopyOnReadDisableCopyOnRead5read_58_disablecopyonread_sequential_1_dense_kernel_1*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp5read_58_disablecopyonread_sequential_1_dense_kernel_1^Read_58/DisableCopyOnRead* 
_output_shapes
:
�b�*
dtype0c
Identity_116IdentityRead_58/ReadVariableOp:value:0*
T0* 
_output_shapes
:
�b�i
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0* 
_output_shapes
:
�b��
Read_59/DisableCopyOnReadDisableCopyOnReadJread_59_disablecopyonread_sequential_1_batch_normalization_3_moving_mean_1*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpJread_59_disablecopyonread_sequential_1_batch_normalization_3_moving_mean_1^Read_59/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_118IdentityRead_59/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_60/DisableCopyOnReadDisableCopyOnReadNread_60_disablecopyonread_sequential_1_batch_normalization_4_moving_variance_1*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOpNread_60_disablecopyonread_sequential_1_batch_normalization_4_moving_variance_1^Read_60/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_120IdentityRead_60/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_61/DisableCopyOnReadDisableCopyOnReadJread_61_disablecopyonread_sequential_1_batch_normalization_1_moving_mean_1*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOpJread_61_disablecopyonread_sequential_1_batch_normalization_1_moving_mean_1^Read_61/DisableCopyOnRead*
_output_shapes
:@*
dtype0]
Identity_122IdentityRead_61/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_62/DisableCopyOnReadDisableCopyOnReadHread_62_disablecopyonread_sequential_1_batch_normalization_moving_mean_1*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOpHread_62_disablecopyonread_sequential_1_batch_normalization_moving_mean_1^Read_62/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_124IdentityRead_62/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_63/DisableCopyOnReadDisableCopyOnReadNread_63_disablecopyonread_sequential_1_batch_normalization_1_moving_variance_1*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOpNread_63_disablecopyonread_sequential_1_batch_normalization_1_moving_variance_1^Read_63/DisableCopyOnRead*
_output_shapes
:@*
dtype0]
Identity_126IdentityRead_63/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_64/DisableCopyOnReadDisableCopyOnReadJread_64_disablecopyonread_sequential_1_batch_normalization_2_moving_mean_1*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOpJread_64_disablecopyonread_sequential_1_batch_normalization_2_moving_mean_1^Read_64/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_128IdentityRead_64/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_65/DisableCopyOnReadDisableCopyOnReadJread_65_disablecopyonread_sequential_1_batch_normalization_4_moving_mean_1*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOpJread_65_disablecopyonread_sequential_1_batch_normalization_4_moving_mean_1^Read_65/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_130IdentityRead_65/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_66/DisableCopyOnReadDisableCopyOnReadLread_66_disablecopyonread_sequential_1_batch_normalization_moving_variance_1*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOpLread_66_disablecopyonread_sequential_1_batch_normalization_moving_variance_1^Read_66/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_132IdentityRead_66/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_67/DisableCopyOnReadDisableCopyOnReadNread_67_disablecopyonread_sequential_1_batch_normalization_2_moving_variance_1*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOpNread_67_disablecopyonread_sequential_1_batch_normalization_2_moving_variance_1^Read_67/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_134IdentityRead_67/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_68/DisableCopyOnReadDisableCopyOnReadNread_68_disablecopyonread_sequential_1_batch_normalization_3_moving_variance_1*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOpNread_68_disablecopyonread_sequential_1_batch_normalization_3_moving_variance_1^Read_68/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_136IdentityRead_68/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes	
:�L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�
value�B�FB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/29/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/30/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/31/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/32/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/33/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:F*
dtype0*�
value�B�FB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *T
dtypesJ
H2F	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_138Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_139IdentityIdentity_138:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_139Identity_139:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=F9

_output_shapes
: 

_user_specified_nameConst:TEP
N
_user_specified_name64sequential_1/batch_normalization_3/moving_variance_1:TDP
N
_user_specified_name64sequential_1/batch_normalization_2/moving_variance_1:RCN
L
_user_specified_name42sequential_1/batch_normalization/moving_variance_1:PBL
J
_user_specified_name20sequential_1/batch_normalization_4/moving_mean_1:PAL
J
_user_specified_name20sequential_1/batch_normalization_2/moving_mean_1:T@P
N
_user_specified_name64sequential_1/batch_normalization_1/moving_variance_1:N?J
H
_user_specified_name0.sequential_1/batch_normalization/moving_mean_1:P>L
J
_user_specified_name20sequential_1/batch_normalization_1/moving_mean_1:T=P
N
_user_specified_name64sequential_1/batch_normalization_4/moving_variance_1:P<L
J
_user_specified_name20sequential_1/batch_normalization_3/moving_mean_1:;;7
5
_user_specified_namesequential_1/dense/kernel_1:<:8
6
_user_specified_namesequential_1/conv2d_4/bias_1:<98
6
_user_specified_namesequential_1/conv2d_3/bias_1:J8F
D
_user_specified_name,*sequential_1/batch_normalization_2/gamma_1:I7E
C
_user_specified_name+)sequential_1/batch_normalization_1/beta_1::66
4
_user_specified_namesequential_1/conv2d/bias_1:I5E
C
_user_specified_name+)sequential_1/batch_normalization_3/beta_1:H4D
B
_user_specified_name*(sequential_1/batch_normalization/gamma_1:935
3
_user_specified_namesequential_1/dense/bias_1:>2:
8
_user_specified_name sequential_1/conv2d_2/kernel_1:<18
6
_user_specified_namesequential_1/conv2d_1/bias_1:=09
7
_user_specified_namesequential_1/dense_1/kernel_1:J/F
D
_user_specified_name,*sequential_1/batch_normalization_4/gamma_1:>.:
8
_user_specified_name sequential_1/conv2d_4/kernel_1:>-:
8
_user_specified_name sequential_1/conv2d_3/kernel_1:;,7
5
_user_specified_namesequential_1/dense_1/bias_1:J+F
D
_user_specified_name,*sequential_1/batch_normalization_1/gamma_1:I*E
C
_user_specified_name+)sequential_1/batch_normalization_4/beta_1:J)F
D
_user_specified_name,*sequential_1/batch_normalization_3/gamma_1:I(E
C
_user_specified_name+)sequential_1/batch_normalization_2/beta_1:<'8
6
_user_specified_namesequential_1/conv2d_2/bias_1:>&:
8
_user_specified_name sequential_1/conv2d_1/kernel_1:<%8
6
_user_specified_namesequential_1/conv2d/kernel_1:G$C
A
_user_specified_name)'sequential_1/batch_normalization/beta_1:(#$
"
_user_specified_name
Variable:*"&
$
_user_specified_name
Variable_1:*!&
$
_user_specified_name
Variable_2:* &
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+'
%
_user_specified_nameVariable_24:+
'
%
_user_specified_nameVariable_25:+	'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_27:+'
%
_user_specified_nameVariable_28:+'
%
_user_specified_nameVariable_29:+'
%
_user_specified_nameVariable_30:+'
%
_user_specified_nameVariable_31:+'
%
_user_specified_nameVariable_32:+'
%
_user_specified_nameVariable_33:+'
%
_user_specified_nameVariable_34:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
��
�%
__inference___call___85138
keras_tensor_4U
;sequential_1_1_conv2d_1_convolution_readvariableop_resource: E
7sequential_1_1_conv2d_1_reshape_readvariableop_resource: O
Asequential_1_1_batch_normalization_1_cast_readvariableop_resource: Q
Csequential_1_1_batch_normalization_1_cast_1_readvariableop_resource: Q
Csequential_1_1_batch_normalization_1_cast_2_readvariableop_resource: Q
Csequential_1_1_batch_normalization_1_cast_3_readvariableop_resource: W
=sequential_1_1_conv2d_1_2_convolution_readvariableop_resource: @G
9sequential_1_1_conv2d_1_2_reshape_readvariableop_resource:@Q
Csequential_1_1_batch_normalization_1_2_cast_readvariableop_resource:@S
Esequential_1_1_batch_normalization_1_2_cast_1_readvariableop_resource:@S
Esequential_1_1_batch_normalization_1_2_cast_2_readvariableop_resource:@S
Esequential_1_1_batch_normalization_1_2_cast_3_readvariableop_resource:@X
=sequential_1_1_conv2d_2_1_convolution_readvariableop_resource:@�H
9sequential_1_1_conv2d_2_1_reshape_readvariableop_resource:	�R
Csequential_1_1_batch_normalization_2_1_cast_readvariableop_resource:	�T
Esequential_1_1_batch_normalization_2_1_cast_1_readvariableop_resource:	�T
Esequential_1_1_batch_normalization_2_1_cast_2_readvariableop_resource:	�T
Esequential_1_1_batch_normalization_2_1_cast_3_readvariableop_resource:	�Y
=sequential_1_1_conv2d_3_1_convolution_readvariableop_resource:��H
9sequential_1_1_conv2d_3_1_reshape_readvariableop_resource:	�R
Csequential_1_1_batch_normalization_3_1_cast_readvariableop_resource:	�T
Esequential_1_1_batch_normalization_3_1_cast_1_readvariableop_resource:	�T
Esequential_1_1_batch_normalization_3_1_cast_2_readvariableop_resource:	�T
Esequential_1_1_batch_normalization_3_1_cast_3_readvariableop_resource:	�Y
=sequential_1_1_conv2d_4_1_convolution_readvariableop_resource:��H
9sequential_1_1_conv2d_4_1_reshape_readvariableop_resource:	�R
Csequential_1_1_batch_normalization_4_1_cast_readvariableop_resource:	�T
Esequential_1_1_batch_normalization_4_1_cast_1_readvariableop_resource:	�T
Esequential_1_1_batch_normalization_4_1_cast_2_readvariableop_resource:	�T
Esequential_1_1_batch_normalization_4_1_cast_3_readvariableop_resource:	�G
3sequential_1_1_dense_1_cast_readvariableop_resource:
�b�E
6sequential_1_1_dense_1_biasadd_readvariableop_resource:	�H
5sequential_1_1_dense_1_2_cast_readvariableop_resource:	�F
8sequential_1_1_dense_1_2_biasadd_readvariableop_resource:
identity��8sequential_1_1/batch_normalization_1/Cast/ReadVariableOp�:sequential_1_1/batch_normalization_1/Cast_1/ReadVariableOp�:sequential_1_1/batch_normalization_1/Cast_2/ReadVariableOp�:sequential_1_1/batch_normalization_1/Cast_3/ReadVariableOp�:sequential_1_1/batch_normalization_1_2/Cast/ReadVariableOp�<sequential_1_1/batch_normalization_1_2/Cast_1/ReadVariableOp�<sequential_1_1/batch_normalization_1_2/Cast_2/ReadVariableOp�<sequential_1_1/batch_normalization_1_2/Cast_3/ReadVariableOp�:sequential_1_1/batch_normalization_2_1/Cast/ReadVariableOp�<sequential_1_1/batch_normalization_2_1/Cast_1/ReadVariableOp�<sequential_1_1/batch_normalization_2_1/Cast_2/ReadVariableOp�<sequential_1_1/batch_normalization_2_1/Cast_3/ReadVariableOp�:sequential_1_1/batch_normalization_3_1/Cast/ReadVariableOp�<sequential_1_1/batch_normalization_3_1/Cast_1/ReadVariableOp�<sequential_1_1/batch_normalization_3_1/Cast_2/ReadVariableOp�<sequential_1_1/batch_normalization_3_1/Cast_3/ReadVariableOp�:sequential_1_1/batch_normalization_4_1/Cast/ReadVariableOp�<sequential_1_1/batch_normalization_4_1/Cast_1/ReadVariableOp�<sequential_1_1/batch_normalization_4_1/Cast_2/ReadVariableOp�<sequential_1_1/batch_normalization_4_1/Cast_3/ReadVariableOp�.sequential_1_1/conv2d_1/Reshape/ReadVariableOp�2sequential_1_1/conv2d_1/convolution/ReadVariableOp�0sequential_1_1/conv2d_1_2/Reshape/ReadVariableOp�4sequential_1_1/conv2d_1_2/convolution/ReadVariableOp�0sequential_1_1/conv2d_2_1/Reshape/ReadVariableOp�4sequential_1_1/conv2d_2_1/convolution/ReadVariableOp�0sequential_1_1/conv2d_3_1/Reshape/ReadVariableOp�4sequential_1_1/conv2d_3_1/convolution/ReadVariableOp�0sequential_1_1/conv2d_4_1/Reshape/ReadVariableOp�4sequential_1_1/conv2d_4_1/convolution/ReadVariableOp�-sequential_1_1/dense_1/BiasAdd/ReadVariableOp�*sequential_1_1/dense_1/Cast/ReadVariableOp�/sequential_1_1/dense_1_2/BiasAdd/ReadVariableOp�,sequential_1_1/dense_1_2/Cast/ReadVariableOp�
2sequential_1_1/conv2d_1/convolution/ReadVariableOpReadVariableOp;sequential_1_1_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
#sequential_1_1/conv2d_1/convolutionConv2Dkeras_tensor_4:sequential_1_1/conv2d_1/convolution/ReadVariableOp:value:0*
T0*1
_output_shapes
:����������� *
paddingSAME*
strides
�
.sequential_1_1/conv2d_1/Reshape/ReadVariableOpReadVariableOp7sequential_1_1_conv2d_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0~
%sequential_1_1/conv2d_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
sequential_1_1/conv2d_1/ReshapeReshape6sequential_1_1/conv2d_1/Reshape/ReadVariableOp:value:0.sequential_1_1/conv2d_1/Reshape/shape:output:0*
T0*&
_output_shapes
: y
sequential_1_1/conv2d_1/SqueezeSqueeze(sequential_1_1/conv2d_1/Reshape:output:0*
T0*
_output_shapes
: �
sequential_1_1/conv2d_1/BiasAddBiasAdd,sequential_1_1/conv2d_1/convolution:output:0(sequential_1_1/conv2d_1/Squeeze:output:0*
T0*1
_output_shapes
:����������� �
sequential_1_1/conv2d_1/ReluRelu(sequential_1_1/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:����������� �
8sequential_1_1/batch_normalization_1/Cast/ReadVariableOpReadVariableOpAsequential_1_1_batch_normalization_1_cast_readvariableop_resource*
_output_shapes
: *
dtype0�
:sequential_1_1/batch_normalization_1/Cast_1/ReadVariableOpReadVariableOpCsequential_1_1_batch_normalization_1_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0�
:sequential_1_1/batch_normalization_1/Cast_2/ReadVariableOpReadVariableOpCsequential_1_1_batch_normalization_1_cast_2_readvariableop_resource*
_output_shapes
: *
dtype0�
:sequential_1_1/batch_normalization_1/Cast_3/ReadVariableOpReadVariableOpCsequential_1_1_batch_normalization_1_cast_3_readvariableop_resource*
_output_shapes
: *
dtype0y
4sequential_1_1/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
2sequential_1_1/batch_normalization_1/batchnorm/addAddV2Bsequential_1_1/batch_normalization_1/Cast_1/ReadVariableOp:value:0=sequential_1_1/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
: �
4sequential_1_1/batch_normalization_1/batchnorm/RsqrtRsqrt6sequential_1_1/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
: �
2sequential_1_1/batch_normalization_1/batchnorm/mulMul8sequential_1_1/batch_normalization_1/batchnorm/Rsqrt:y:0Bsequential_1_1/batch_normalization_1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes
: �
4sequential_1_1/batch_normalization_1/batchnorm/mul_1Mul*sequential_1_1/conv2d_1/Relu:activations:06sequential_1_1/batch_normalization_1/batchnorm/mul:z:0*
T0*1
_output_shapes
:����������� �
4sequential_1_1/batch_normalization_1/batchnorm/mul_2Mul@sequential_1_1/batch_normalization_1/Cast/ReadVariableOp:value:06sequential_1_1/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
: �
2sequential_1_1/batch_normalization_1/batchnorm/subSubBsequential_1_1/batch_normalization_1/Cast_3/ReadVariableOp:value:08sequential_1_1/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: �
4sequential_1_1/batch_normalization_1/batchnorm/add_1AddV28sequential_1_1/batch_normalization_1/batchnorm/mul_1:z:06sequential_1_1/batch_normalization_1/batchnorm/sub:z:0*
T0*1
_output_shapes
:����������� �
(sequential_1_1/max_pooling2d_1/MaxPool2dMaxPool8sequential_1_1/batch_normalization_1/batchnorm/add_1:z:0*/
_output_shapes
:���������pp *
ksize
*
paddingVALID*
strides
�
4sequential_1_1/conv2d_1_2/convolution/ReadVariableOpReadVariableOp=sequential_1_1_conv2d_1_2_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
%sequential_1_1/conv2d_1_2/convolutionConv2D1sequential_1_1/max_pooling2d_1/MaxPool2d:output:0<sequential_1_1/conv2d_1_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������pp@*
paddingSAME*
strides
�
0sequential_1_1/conv2d_1_2/Reshape/ReadVariableOpReadVariableOp9sequential_1_1_conv2d_1_2_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0�
'sequential_1_1/conv2d_1_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
!sequential_1_1/conv2d_1_2/ReshapeReshape8sequential_1_1/conv2d_1_2/Reshape/ReadVariableOp:value:00sequential_1_1/conv2d_1_2/Reshape/shape:output:0*
T0*&
_output_shapes
:@}
!sequential_1_1/conv2d_1_2/SqueezeSqueeze*sequential_1_1/conv2d_1_2/Reshape:output:0*
T0*
_output_shapes
:@�
!sequential_1_1/conv2d_1_2/BiasAddBiasAdd.sequential_1_1/conv2d_1_2/convolution:output:0*sequential_1_1/conv2d_1_2/Squeeze:output:0*
T0*/
_output_shapes
:���������pp@�
sequential_1_1/conv2d_1_2/ReluRelu*sequential_1_1/conv2d_1_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������pp@�
:sequential_1_1/batch_normalization_1_2/Cast/ReadVariableOpReadVariableOpCsequential_1_1_batch_normalization_1_2_cast_readvariableop_resource*
_output_shapes
:@*
dtype0�
<sequential_1_1/batch_normalization_1_2/Cast_1/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_1_2_cast_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
<sequential_1_1/batch_normalization_1_2/Cast_2/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_1_2_cast_2_readvariableop_resource*
_output_shapes
:@*
dtype0�
<sequential_1_1/batch_normalization_1_2/Cast_3/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_1_2_cast_3_readvariableop_resource*
_output_shapes
:@*
dtype0{
6sequential_1_1/batch_normalization_1_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4sequential_1_1/batch_normalization_1_2/batchnorm/addAddV2Dsequential_1_1/batch_normalization_1_2/Cast_1/ReadVariableOp:value:0?sequential_1_1/batch_normalization_1_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
6sequential_1_1/batch_normalization_1_2/batchnorm/RsqrtRsqrt8sequential_1_1/batch_normalization_1_2/batchnorm/add:z:0*
T0*
_output_shapes
:@�
4sequential_1_1/batch_normalization_1_2/batchnorm/mulMul:sequential_1_1/batch_normalization_1_2/batchnorm/Rsqrt:y:0Dsequential_1_1/batch_normalization_1_2/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
6sequential_1_1/batch_normalization_1_2/batchnorm/mul_1Mul,sequential_1_1/conv2d_1_2/Relu:activations:08sequential_1_1/batch_normalization_1_2/batchnorm/mul:z:0*
T0*/
_output_shapes
:���������pp@�
6sequential_1_1/batch_normalization_1_2/batchnorm/mul_2MulBsequential_1_1/batch_normalization_1_2/Cast/ReadVariableOp:value:08sequential_1_1/batch_normalization_1_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
4sequential_1_1/batch_normalization_1_2/batchnorm/subSubDsequential_1_1/batch_normalization_1_2/Cast_3/ReadVariableOp:value:0:sequential_1_1/batch_normalization_1_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
6sequential_1_1/batch_normalization_1_2/batchnorm/add_1AddV2:sequential_1_1/batch_normalization_1_2/batchnorm/mul_1:z:08sequential_1_1/batch_normalization_1_2/batchnorm/sub:z:0*
T0*/
_output_shapes
:���������pp@�
*sequential_1_1/max_pooling2d_1_2/MaxPool2dMaxPool:sequential_1_1/batch_normalization_1_2/batchnorm/add_1:z:0*/
_output_shapes
:���������88@*
ksize
*
paddingVALID*
strides
�
4sequential_1_1/conv2d_2_1/convolution/ReadVariableOpReadVariableOp=sequential_1_1_conv2d_2_1_convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
%sequential_1_1/conv2d_2_1/convolutionConv2D3sequential_1_1/max_pooling2d_1_2/MaxPool2d:output:0<sequential_1_1/conv2d_2_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
�
0sequential_1_1/conv2d_2_1/Reshape/ReadVariableOpReadVariableOp9sequential_1_1_conv2d_2_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential_1_1/conv2d_2_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
!sequential_1_1/conv2d_2_1/ReshapeReshape8sequential_1_1/conv2d_2_1/Reshape/ReadVariableOp:value:00sequential_1_1/conv2d_2_1/Reshape/shape:output:0*
T0*'
_output_shapes
:�~
!sequential_1_1/conv2d_2_1/SqueezeSqueeze*sequential_1_1/conv2d_2_1/Reshape:output:0*
T0*
_output_shapes	
:��
!sequential_1_1/conv2d_2_1/BiasAddBiasAdd.sequential_1_1/conv2d_2_1/convolution:output:0*sequential_1_1/conv2d_2_1/Squeeze:output:0*
T0*0
_output_shapes
:���������88��
sequential_1_1/conv2d_2_1/ReluRelu*sequential_1_1/conv2d_2_1/BiasAdd:output:0*
T0*0
_output_shapes
:���������88��
:sequential_1_1/batch_normalization_2_1/Cast/ReadVariableOpReadVariableOpCsequential_1_1_batch_normalization_2_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<sequential_1_1/batch_normalization_2_1/Cast_1/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_2_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<sequential_1_1/batch_normalization_2_1/Cast_2/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_2_1_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<sequential_1_1/batch_normalization_2_1/Cast_3/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_2_1_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0{
6sequential_1_1/batch_normalization_2_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4sequential_1_1/batch_normalization_2_1/batchnorm/addAddV2Dsequential_1_1/batch_normalization_2_1/Cast_1/ReadVariableOp:value:0?sequential_1_1/batch_normalization_2_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
6sequential_1_1/batch_normalization_2_1/batchnorm/RsqrtRsqrt8sequential_1_1/batch_normalization_2_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
4sequential_1_1/batch_normalization_2_1/batchnorm/mulMul:sequential_1_1/batch_normalization_2_1/batchnorm/Rsqrt:y:0Dsequential_1_1/batch_normalization_2_1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
6sequential_1_1/batch_normalization_2_1/batchnorm/mul_1Mul,sequential_1_1/conv2d_2_1/Relu:activations:08sequential_1_1/batch_normalization_2_1/batchnorm/mul:z:0*
T0*0
_output_shapes
:���������88��
6sequential_1_1/batch_normalization_2_1/batchnorm/mul_2MulBsequential_1_1/batch_normalization_2_1/Cast/ReadVariableOp:value:08sequential_1_1/batch_normalization_2_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
4sequential_1_1/batch_normalization_2_1/batchnorm/subSubDsequential_1_1/batch_normalization_2_1/Cast_3/ReadVariableOp:value:0:sequential_1_1/batch_normalization_2_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
6sequential_1_1/batch_normalization_2_1/batchnorm/add_1AddV2:sequential_1_1/batch_normalization_2_1/batchnorm/mul_1:z:08sequential_1_1/batch_normalization_2_1/batchnorm/sub:z:0*
T0*0
_output_shapes
:���������88��
*sequential_1_1/max_pooling2d_2_1/MaxPool2dMaxPool:sequential_1_1/batch_normalization_2_1/batchnorm/add_1:z:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
4sequential_1_1/conv2d_3_1/convolution/ReadVariableOpReadVariableOp=sequential_1_1_conv2d_3_1_convolution_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%sequential_1_1/conv2d_3_1/convolutionConv2D3sequential_1_1/max_pooling2d_2_1/MaxPool2d:output:0<sequential_1_1/conv2d_3_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
0sequential_1_1/conv2d_3_1/Reshape/ReadVariableOpReadVariableOp9sequential_1_1_conv2d_3_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential_1_1/conv2d_3_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
!sequential_1_1/conv2d_3_1/ReshapeReshape8sequential_1_1/conv2d_3_1/Reshape/ReadVariableOp:value:00sequential_1_1/conv2d_3_1/Reshape/shape:output:0*
T0*'
_output_shapes
:�~
!sequential_1_1/conv2d_3_1/SqueezeSqueeze*sequential_1_1/conv2d_3_1/Reshape:output:0*
T0*
_output_shapes	
:��
!sequential_1_1/conv2d_3_1/BiasAddBiasAdd.sequential_1_1/conv2d_3_1/convolution:output:0*sequential_1_1/conv2d_3_1/Squeeze:output:0*
T0*0
_output_shapes
:�����������
sequential_1_1/conv2d_3_1/ReluRelu*sequential_1_1/conv2d_3_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
:sequential_1_1/batch_normalization_3_1/Cast/ReadVariableOpReadVariableOpCsequential_1_1_batch_normalization_3_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<sequential_1_1/batch_normalization_3_1/Cast_1/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_3_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<sequential_1_1/batch_normalization_3_1/Cast_2/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_3_1_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<sequential_1_1/batch_normalization_3_1/Cast_3/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_3_1_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0{
6sequential_1_1/batch_normalization_3_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4sequential_1_1/batch_normalization_3_1/batchnorm/addAddV2Dsequential_1_1/batch_normalization_3_1/Cast_1/ReadVariableOp:value:0?sequential_1_1/batch_normalization_3_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
6sequential_1_1/batch_normalization_3_1/batchnorm/RsqrtRsqrt8sequential_1_1/batch_normalization_3_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
4sequential_1_1/batch_normalization_3_1/batchnorm/mulMul:sequential_1_1/batch_normalization_3_1/batchnorm/Rsqrt:y:0Dsequential_1_1/batch_normalization_3_1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
6sequential_1_1/batch_normalization_3_1/batchnorm/mul_1Mul,sequential_1_1/conv2d_3_1/Relu:activations:08sequential_1_1/batch_normalization_3_1/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
6sequential_1_1/batch_normalization_3_1/batchnorm/mul_2MulBsequential_1_1/batch_normalization_3_1/Cast/ReadVariableOp:value:08sequential_1_1/batch_normalization_3_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
4sequential_1_1/batch_normalization_3_1/batchnorm/subSubDsequential_1_1/batch_normalization_3_1/Cast_3/ReadVariableOp:value:0:sequential_1_1/batch_normalization_3_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
6sequential_1_1/batch_normalization_3_1/batchnorm/add_1AddV2:sequential_1_1/batch_normalization_3_1/batchnorm/mul_1:z:08sequential_1_1/batch_normalization_3_1/batchnorm/sub:z:0*
T0*0
_output_shapes
:�����������
*sequential_1_1/max_pooling2d_3_1/MaxPool2dMaxPool:sequential_1_1/batch_normalization_3_1/batchnorm/add_1:z:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
4sequential_1_1/conv2d_4_1/convolution/ReadVariableOpReadVariableOp=sequential_1_1_conv2d_4_1_convolution_readvariableop_resource*(
_output_shapes
:��*
dtype0�
%sequential_1_1/conv2d_4_1/convolutionConv2D3sequential_1_1/max_pooling2d_3_1/MaxPool2d:output:0<sequential_1_1/conv2d_4_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
0sequential_1_1/conv2d_4_1/Reshape/ReadVariableOpReadVariableOp9sequential_1_1_conv2d_4_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'sequential_1_1/conv2d_4_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
!sequential_1_1/conv2d_4_1/ReshapeReshape8sequential_1_1/conv2d_4_1/Reshape/ReadVariableOp:value:00sequential_1_1/conv2d_4_1/Reshape/shape:output:0*
T0*'
_output_shapes
:�~
!sequential_1_1/conv2d_4_1/SqueezeSqueeze*sequential_1_1/conv2d_4_1/Reshape:output:0*
T0*
_output_shapes	
:��
!sequential_1_1/conv2d_4_1/BiasAddBiasAdd.sequential_1_1/conv2d_4_1/convolution:output:0*sequential_1_1/conv2d_4_1/Squeeze:output:0*
T0*0
_output_shapes
:�����������
sequential_1_1/conv2d_4_1/ReluRelu*sequential_1_1/conv2d_4_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
:sequential_1_1/batch_normalization_4_1/Cast/ReadVariableOpReadVariableOpCsequential_1_1_batch_normalization_4_1_cast_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<sequential_1_1/batch_normalization_4_1/Cast_1/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_4_1_cast_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<sequential_1_1/batch_normalization_4_1/Cast_2/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_4_1_cast_2_readvariableop_resource*
_output_shapes	
:�*
dtype0�
<sequential_1_1/batch_normalization_4_1/Cast_3/ReadVariableOpReadVariableOpEsequential_1_1_batch_normalization_4_1_cast_3_readvariableop_resource*
_output_shapes	
:�*
dtype0{
6sequential_1_1/batch_normalization_4_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4sequential_1_1/batch_normalization_4_1/batchnorm/addAddV2Dsequential_1_1/batch_normalization_4_1/Cast_1/ReadVariableOp:value:0?sequential_1_1/batch_normalization_4_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
6sequential_1_1/batch_normalization_4_1/batchnorm/RsqrtRsqrt8sequential_1_1/batch_normalization_4_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
4sequential_1_1/batch_normalization_4_1/batchnorm/mulMul:sequential_1_1/batch_normalization_4_1/batchnorm/Rsqrt:y:0Dsequential_1_1/batch_normalization_4_1/Cast_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
6sequential_1_1/batch_normalization_4_1/batchnorm/mul_1Mul,sequential_1_1/conv2d_4_1/Relu:activations:08sequential_1_1/batch_normalization_4_1/batchnorm/mul:z:0*
T0*0
_output_shapes
:�����������
6sequential_1_1/batch_normalization_4_1/batchnorm/mul_2MulBsequential_1_1/batch_normalization_4_1/Cast/ReadVariableOp:value:08sequential_1_1/batch_normalization_4_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
4sequential_1_1/batch_normalization_4_1/batchnorm/subSubDsequential_1_1/batch_normalization_4_1/Cast_3/ReadVariableOp:value:0:sequential_1_1/batch_normalization_4_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
6sequential_1_1/batch_normalization_4_1/batchnorm/add_1AddV2:sequential_1_1/batch_normalization_4_1/batchnorm/mul_1:z:08sequential_1_1/batch_normalization_4_1/batchnorm/sub:z:0*
T0*0
_output_shapes
:�����������
*sequential_1_1/max_pooling2d_4_1/MaxPool2dMaxPool:sequential_1_1/batch_normalization_4_1/batchnorm/add_1:z:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
w
&sequential_1_1/flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"���� 1  �
 sequential_1_1/flatten_1/ReshapeReshape3sequential_1_1/max_pooling2d_4_1/MaxPool2d:output:0/sequential_1_1/flatten_1/Reshape/shape:output:0*
T0*(
_output_shapes
:����������b�
*sequential_1_1/dense_1/Cast/ReadVariableOpReadVariableOp3sequential_1_1_dense_1_cast_readvariableop_resource* 
_output_shapes
:
�b�*
dtype0�
sequential_1_1/dense_1/MatMulMatMul)sequential_1_1/flatten_1/Reshape:output:02sequential_1_1/dense_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_1_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp6sequential_1_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_1_1/dense_1/BiasAddBiasAdd'sequential_1_1/dense_1/MatMul:product:05sequential_1_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_1_1/dense_1/ReluRelu'sequential_1_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
,sequential_1_1/dense_1_2/Cast/ReadVariableOpReadVariableOp5sequential_1_1_dense_1_2_cast_readvariableop_resource*
_output_shapes
:	�*
dtype0�
sequential_1_1/dense_1_2/MatMulMatMul)sequential_1_1/dense_1/Relu:activations:04sequential_1_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/sequential_1_1/dense_1_2/BiasAdd/ReadVariableOpReadVariableOp8sequential_1_1_dense_1_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
 sequential_1_1/dense_1_2/BiasAddBiasAdd)sequential_1_1/dense_1_2/MatMul:product:07sequential_1_1/dense_1_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 sequential_1_1/dense_1_2/SoftmaxSoftmax)sequential_1_1/dense_1_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������y
IdentityIdentity*sequential_1_1/dense_1_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp9^sequential_1_1/batch_normalization_1/Cast/ReadVariableOp;^sequential_1_1/batch_normalization_1/Cast_1/ReadVariableOp;^sequential_1_1/batch_normalization_1/Cast_2/ReadVariableOp;^sequential_1_1/batch_normalization_1/Cast_3/ReadVariableOp;^sequential_1_1/batch_normalization_1_2/Cast/ReadVariableOp=^sequential_1_1/batch_normalization_1_2/Cast_1/ReadVariableOp=^sequential_1_1/batch_normalization_1_2/Cast_2/ReadVariableOp=^sequential_1_1/batch_normalization_1_2/Cast_3/ReadVariableOp;^sequential_1_1/batch_normalization_2_1/Cast/ReadVariableOp=^sequential_1_1/batch_normalization_2_1/Cast_1/ReadVariableOp=^sequential_1_1/batch_normalization_2_1/Cast_2/ReadVariableOp=^sequential_1_1/batch_normalization_2_1/Cast_3/ReadVariableOp;^sequential_1_1/batch_normalization_3_1/Cast/ReadVariableOp=^sequential_1_1/batch_normalization_3_1/Cast_1/ReadVariableOp=^sequential_1_1/batch_normalization_3_1/Cast_2/ReadVariableOp=^sequential_1_1/batch_normalization_3_1/Cast_3/ReadVariableOp;^sequential_1_1/batch_normalization_4_1/Cast/ReadVariableOp=^sequential_1_1/batch_normalization_4_1/Cast_1/ReadVariableOp=^sequential_1_1/batch_normalization_4_1/Cast_2/ReadVariableOp=^sequential_1_1/batch_normalization_4_1/Cast_3/ReadVariableOp/^sequential_1_1/conv2d_1/Reshape/ReadVariableOp3^sequential_1_1/conv2d_1/convolution/ReadVariableOp1^sequential_1_1/conv2d_1_2/Reshape/ReadVariableOp5^sequential_1_1/conv2d_1_2/convolution/ReadVariableOp1^sequential_1_1/conv2d_2_1/Reshape/ReadVariableOp5^sequential_1_1/conv2d_2_1/convolution/ReadVariableOp1^sequential_1_1/conv2d_3_1/Reshape/ReadVariableOp5^sequential_1_1/conv2d_3_1/convolution/ReadVariableOp1^sequential_1_1/conv2d_4_1/Reshape/ReadVariableOp5^sequential_1_1/conv2d_4_1/convolution/ReadVariableOp.^sequential_1_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1_1/dense_1/Cast/ReadVariableOp0^sequential_1_1/dense_1_2/BiasAdd/ReadVariableOp-^sequential_1_1/dense_1_2/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8sequential_1_1/batch_normalization_1/Cast/ReadVariableOp8sequential_1_1/batch_normalization_1/Cast/ReadVariableOp2x
:sequential_1_1/batch_normalization_1/Cast_1/ReadVariableOp:sequential_1_1/batch_normalization_1/Cast_1/ReadVariableOp2x
:sequential_1_1/batch_normalization_1/Cast_2/ReadVariableOp:sequential_1_1/batch_normalization_1/Cast_2/ReadVariableOp2x
:sequential_1_1/batch_normalization_1/Cast_3/ReadVariableOp:sequential_1_1/batch_normalization_1/Cast_3/ReadVariableOp2x
:sequential_1_1/batch_normalization_1_2/Cast/ReadVariableOp:sequential_1_1/batch_normalization_1_2/Cast/ReadVariableOp2|
<sequential_1_1/batch_normalization_1_2/Cast_1/ReadVariableOp<sequential_1_1/batch_normalization_1_2/Cast_1/ReadVariableOp2|
<sequential_1_1/batch_normalization_1_2/Cast_2/ReadVariableOp<sequential_1_1/batch_normalization_1_2/Cast_2/ReadVariableOp2|
<sequential_1_1/batch_normalization_1_2/Cast_3/ReadVariableOp<sequential_1_1/batch_normalization_1_2/Cast_3/ReadVariableOp2x
:sequential_1_1/batch_normalization_2_1/Cast/ReadVariableOp:sequential_1_1/batch_normalization_2_1/Cast/ReadVariableOp2|
<sequential_1_1/batch_normalization_2_1/Cast_1/ReadVariableOp<sequential_1_1/batch_normalization_2_1/Cast_1/ReadVariableOp2|
<sequential_1_1/batch_normalization_2_1/Cast_2/ReadVariableOp<sequential_1_1/batch_normalization_2_1/Cast_2/ReadVariableOp2|
<sequential_1_1/batch_normalization_2_1/Cast_3/ReadVariableOp<sequential_1_1/batch_normalization_2_1/Cast_3/ReadVariableOp2x
:sequential_1_1/batch_normalization_3_1/Cast/ReadVariableOp:sequential_1_1/batch_normalization_3_1/Cast/ReadVariableOp2|
<sequential_1_1/batch_normalization_3_1/Cast_1/ReadVariableOp<sequential_1_1/batch_normalization_3_1/Cast_1/ReadVariableOp2|
<sequential_1_1/batch_normalization_3_1/Cast_2/ReadVariableOp<sequential_1_1/batch_normalization_3_1/Cast_2/ReadVariableOp2|
<sequential_1_1/batch_normalization_3_1/Cast_3/ReadVariableOp<sequential_1_1/batch_normalization_3_1/Cast_3/ReadVariableOp2x
:sequential_1_1/batch_normalization_4_1/Cast/ReadVariableOp:sequential_1_1/batch_normalization_4_1/Cast/ReadVariableOp2|
<sequential_1_1/batch_normalization_4_1/Cast_1/ReadVariableOp<sequential_1_1/batch_normalization_4_1/Cast_1/ReadVariableOp2|
<sequential_1_1/batch_normalization_4_1/Cast_2/ReadVariableOp<sequential_1_1/batch_normalization_4_1/Cast_2/ReadVariableOp2|
<sequential_1_1/batch_normalization_4_1/Cast_3/ReadVariableOp<sequential_1_1/batch_normalization_4_1/Cast_3/ReadVariableOp2`
.sequential_1_1/conv2d_1/Reshape/ReadVariableOp.sequential_1_1/conv2d_1/Reshape/ReadVariableOp2h
2sequential_1_1/conv2d_1/convolution/ReadVariableOp2sequential_1_1/conv2d_1/convolution/ReadVariableOp2d
0sequential_1_1/conv2d_1_2/Reshape/ReadVariableOp0sequential_1_1/conv2d_1_2/Reshape/ReadVariableOp2l
4sequential_1_1/conv2d_1_2/convolution/ReadVariableOp4sequential_1_1/conv2d_1_2/convolution/ReadVariableOp2d
0sequential_1_1/conv2d_2_1/Reshape/ReadVariableOp0sequential_1_1/conv2d_2_1/Reshape/ReadVariableOp2l
4sequential_1_1/conv2d_2_1/convolution/ReadVariableOp4sequential_1_1/conv2d_2_1/convolution/ReadVariableOp2d
0sequential_1_1/conv2d_3_1/Reshape/ReadVariableOp0sequential_1_1/conv2d_3_1/Reshape/ReadVariableOp2l
4sequential_1_1/conv2d_3_1/convolution/ReadVariableOp4sequential_1_1/conv2d_3_1/convolution/ReadVariableOp2d
0sequential_1_1/conv2d_4_1/Reshape/ReadVariableOp0sequential_1_1/conv2d_4_1/Reshape/ReadVariableOp2l
4sequential_1_1/conv2d_4_1/convolution/ReadVariableOp4sequential_1_1/conv2d_4_1/convolution/ReadVariableOp2^
-sequential_1_1/dense_1/BiasAdd/ReadVariableOp-sequential_1_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1_1/dense_1/Cast/ReadVariableOp*sequential_1_1/dense_1/Cast/ReadVariableOp2b
/sequential_1_1/dense_1_2/BiasAdd/ReadVariableOp/sequential_1_1/dense_1_2/BiasAdd/ReadVariableOp2\
,sequential_1_1/dense_1_2/Cast/ReadVariableOp,sequential_1_1/dense_1_2/Cast/ReadVariableOp:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:a ]
1
_output_shapes
:�����������
(
_user_specified_namekeras_tensor_4"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
I
keras_tensor_47
serve_keras_tensor_4:0�����������<
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict*�
serving_default�
S
keras_tensor_4A
 serving_default_keras_tensor_4:0�����������>
output_02
StatefulPartitionedCall_1:0���������tensorflow/serving/predict:�2
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30
'31
(32
)33
*34"
trackable_list_wrapper
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
 16
!17
"18
#19
&20
'21
)22
*23"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
$8
%9
(10"
trackable_list_wrapper
�
+0
,1
-2
.3
/4
05
16
27
38
49
510
611
712
813
914
:15
;16
<17
=18
>19
?20
@21
A22
B23
C24
D25
E26
F27
G28
H29
I30
J31
K32
L33"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mtrace_02�
__inference___call___85138�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/
keras_tensor_4�����������zMtrace_0
7
	Nserve
Oserving_default"
signature_map
4:2 2sequential_1/conv2d/kernel
&:$ 2sequential_1/conv2d/bias
4:2 2&sequential_1/batch_normalization/gamma
3:1 2%sequential_1/batch_normalization/beta
8:6 2,sequential_1/batch_normalization/moving_mean
<:: 20sequential_1/batch_normalization/moving_variance
6:4 @2sequential_1/conv2d_1/kernel
(:&@2sequential_1/conv2d_1/bias
6:4@2(sequential_1/batch_normalization_1/gamma
5:3@2'sequential_1/batch_normalization_1/beta
::8@2.sequential_1/batch_normalization_1/moving_mean
>:<@22sequential_1/batch_normalization_1/moving_variance
7:5@�2sequential_1/conv2d_2/kernel
):'�2sequential_1/conv2d_2/bias
7:5�2(sequential_1/batch_normalization_2/gamma
6:4�2'sequential_1/batch_normalization_2/beta
;:9�2.sequential_1/batch_normalization_2/moving_mean
?:=�22sequential_1/batch_normalization_2/moving_variance
8:6��2sequential_1/conv2d_3/kernel
):'�2sequential_1/conv2d_3/bias
7:5�2(sequential_1/batch_normalization_3/gamma
6:4�2'sequential_1/batch_normalization_3/beta
;:9�2.sequential_1/batch_normalization_3/moving_mean
?:=�22sequential_1/batch_normalization_3/moving_variance
8:6��2sequential_1/conv2d_4/kernel
):'�2sequential_1/conv2d_4/bias
7:5�2(sequential_1/batch_normalization_4/gamma
6:4�2'sequential_1/batch_normalization_4/beta
;:9�2.sequential_1/batch_normalization_4/moving_mean
?:=�22sequential_1/batch_normalization_4/moving_variance
-:+
�b�2sequential_1/dense/kernel
&:$�2sequential_1/dense/bias
1:/	2%seed_generator_3/seed_generator_state
.:,	�2sequential_1/dense_1/kernel
':%2sequential_1/dense_1/bias
3:1 2%sequential_1/batch_normalization/beta
4:2 2sequential_1/conv2d/kernel
6:4 @2sequential_1/conv2d_1/kernel
):'�2sequential_1/conv2d_2/bias
6:4�2'sequential_1/batch_normalization_2/beta
7:5�2(sequential_1/batch_normalization_3/gamma
6:4�2'sequential_1/batch_normalization_4/beta
6:4@2(sequential_1/batch_normalization_1/gamma
':%2sequential_1/dense_1/bias
8:6��2sequential_1/conv2d_3/kernel
8:6��2sequential_1/conv2d_4/kernel
7:5�2(sequential_1/batch_normalization_4/gamma
.:,	�2sequential_1/dense_1/kernel
(:&@2sequential_1/conv2d_1/bias
7:5@�2sequential_1/conv2d_2/kernel
&:$�2sequential_1/dense/bias
4:2 2&sequential_1/batch_normalization/gamma
6:4�2'sequential_1/batch_normalization_3/beta
&:$ 2sequential_1/conv2d/bias
5:3@2'sequential_1/batch_normalization_1/beta
7:5�2(sequential_1/batch_normalization_2/gamma
):'�2sequential_1/conv2d_3/bias
):'�2sequential_1/conv2d_4/bias
-:+
�b�2sequential_1/dense/kernel
;:9�2.sequential_1/batch_normalization_3/moving_mean
?:=�22sequential_1/batch_normalization_4/moving_variance
::8@2.sequential_1/batch_normalization_1/moving_mean
8:6 2,sequential_1/batch_normalization/moving_mean
>:<@22sequential_1/batch_normalization_1/moving_variance
;:9�2.sequential_1/batch_normalization_2/moving_mean
;:9�2.sequential_1/batch_normalization_4/moving_mean
<:: 20sequential_1/batch_normalization/moving_variance
?:=�22sequential_1/batch_normalization_2/moving_variance
?:=�22sequential_1/batch_normalization_3/moving_variance
�B�
__inference___call___85138keras_tensor_4"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
,__inference_signature_wrapper___call___85212keras_tensor_4"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 #

kwonlyargs�
jkeras_tensor_4
kwonlydefaults
 
annotations� *
 
�B�
,__inference_signature_wrapper___call___85285keras_tensor_4"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 #

kwonlyargs�
jkeras_tensor_4
kwonlydefaults
 
annotations� *
 �
__inference___call___85138�"	
 !$%"#&')*A�>
7�4
2�/
keras_tensor_4�����������
� "!�
unknown����������
,__inference_signature_wrapper___call___85212�"	
 !$%"#&')*S�P
� 
I�F
D
keras_tensor_42�/
keras_tensor_4�����������"3�0
.
output_0"�
output_0����������
,__inference_signature_wrapper___call___85285�"	
 !$%"#&')*S�P
� 
I�F
D
keras_tensor_42�/
keras_tensor_4�����������"3�0
.
output_0"�
output_0���������