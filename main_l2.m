clear
clc
setenv('SNOPT_LICENSE','D:\APPs\MATLAB\snopt7.lic')

total_size=270;%(要根据数据更改)
Size=54;%(要根据数据更改)
[original_label, original_inst] =  libsvmread(['heart.txt']);%(要根据数据更改)

heart_scale_label=original_label(1:total_size);
heart_scale_inst=original_inst(1:total_size,:);
feature=size(heart_scale_inst,2);

%% 不是正负1
if isempty(find(heart_scale_label==1)) %1: no
    if isempty(find(heart_scale_label==-1)) %1: no, -1: no
        heart_scale_label(heart_scale_label==heart_scale_label(1)) = 1; %把==first的改成1
        heart_scale_label(heart_scale_label~=1) = -1; %把此时不==1的改成-1
    else %1: no, -1: yes
        heart_scale_label(heart_scale_label~=-1) = 1; %把~=-1的改成1
    end
else %1: yes
    if isempty(find(heart_scale_label==-1)) %1: yes, -1: no
        heart_scale_label(heart_scale_label~=1) = -1; %把~=1的改成-1
    end
end

%%
tic
m_1= Size;%三折交叉验证中1份的数据集大小（验证集）
m_2=Size*2; %三折交叉验证中2份的数据集大小（训练集）
n_tr=Size*3;
n_te=total_size-n_tr;
%如果这个数据集就只包含l_train.
h_lab_1=heart_scale_label(1:Size);%第一份的y，是三折的验证集的y
rep_3=repmat(h_lab_1,1,feature);
h_lab_2=heart_scale_label(Size+1:2*Size);%第二份的y, 是二折的验证集的y
rep_2=repmat(h_lab_2,1, feature);
h_lab_3= heart_scale_label(2*Size+1:3*Size);%第三份的y，是一折的验证集的y
rep_1=repmat(h_lab_3,1, feature);
h_inst_1=heart_scale_inst(1:Size,:); %是三折的验证集的x
h_inst_2=heart_scale_inst(Size+1: 2*Size,:); %是二折的验证集的x
h_inst_3=heart_scale_inst(2*Size+1: 3*Size,:); %是一折的验证集的x
comb_lab_12=[h_lab_1; h_lab_2];%一折的训练集的y
comb_inst_12=[ h_inst_1; h_inst_2];% 一折的训练集的x
comb_lab_13=[h_lab_1; h_lab_3];%二折的训练集的y
comb_inst_13=[ h_inst_1; h_inst_3];% 二折的训练集的x
comb_lab_23=[h_lab_2; h_lab_3];%三折的训练集的y
comb_inst_23=[ h_inst_2; h_inst_3]; %三折的训练集的x
rep_12= repmat(comb_lab_12,1,feature);
rep_13= repmat(comb_lab_13,1,feature);
rep_23= repmat(comb_lab_23,1,feature);
B_1=rep_12.*comb_inst_12;
A_1=rep_1.* h_inst_3;
B_2=rep_13.*comb_inst_13;
A_2=rep_2.* h_inst_2;
B_3=rep_23.* comb_inst_23;
A_3=rep_3.* h_inst_1;
Tf=3;%T_fold
Tfm1=Tf*m_1;
Tfm2=Tf*m_2;
m_hat=Tf*m_1+2*Tf*m_2;%C之外变量的个数
A=[A_1,zeros(m_1,2*feature);zeros(m_1,feature),A_2, zeros(m_1,feature); zeros(m_1,2*feature),A_3];
B=[B_1,zeros(m_2, 2*feature);zeros(m_2, feature),B_2, zeros(m_2,feature);zeros(m_2,2*feature),B_3];
AB=A*B';
BB=B*B';
SP_1_Line2=[zeros(Tfm2,1),zeros(Tfm2,Tfm1),-BB,-eye(Tfm2)];
SP_1_Line1=[zeros(Tfm1,1),-eye(Tfm1),-AB,zeros(Tfm1,Tfm2)];
SP_1=[SP_1_Line1,zeros(Tfm1,1);SP_1_Line2,zeros(Tfm2,1)];
cd_1=-ones(Tfm1+Tfm2,1);

%% 设置参数
lb_one=0;
lb=[lb_one;zeros(m_hat,1);-inf];
v_01=1;
v_0=[v_01;zeros(m_hat,1);1];%最后一个分量代表t_k的初始值t_0
[v,f,info,output,lambda,states]=snsolve(@(v_0)obj_l2(v_0),v_0,SP_1,cd_1,[],[],lb,[],@(v_0)nonlinearcons_l2(v_0,Size,feature,heart_scale_label,heart_scale_inst));
iter=output.iterations;
C= v(1)*1.5;
t=toc;
% G_opt=[-SP_1_Line2*v(1:m_hat+1)-ones(Tfm2,1);2*v(1)*v(Tfm1+Tfm2+2:m_hat+1)-v(Tfm1+2:Tfm1+Tfm2+1)];
% H_opt=v(Tfm1+2:m_hat+1);
% GH=G_opt-H_opt;
% index_1=find(GH>0);
% index_2=find(GH<=0);
% absH=abs(H_opt(index_1));
% absG=abs(G_opt(index_2));
% combine=[absH;absG];
% maxVio=max(combine);

%% 测试结果
Xtrain =  heart_scale_inst(1:n_tr,:);
Ytrain = heart_scale_label(1:n_tr);
Xtest  =  heart_scale_inst(n_tr+1:end,:);
Ytest  = heart_scale_label(n_tr+1:end);

% Training
model = SemismoothNewtonTrain(Xtrain', Ytrain, C);
%% Testing
predicting_label = SemismoothNewtonPredict(model, Xtest');
error_rate = mean(predicting_label ~= Ytest)*100;
fprintf('==== Finish Training! ====\n')
fprintf('The error rate of the test set : %.3f\n', error_rate)
fprintf('Time : %.3f\n', t)
fprintf('The final hyperparameter : %.4f\n', C)


