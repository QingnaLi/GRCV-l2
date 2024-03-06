function [c,ceq,dc,dceq]=nonlinearcons_l2(v,Size,feature,heart_scale_label,heart_scale_inst)

m_1= Size;%���۽�����֤��1�ݵ����ݼ���С����֤����
m_2=Size*2; %���۽�����֤��2�ݵ����ݼ���С��ѵ������

%%
h_lab_1=heart_scale_label(1:Size);%��һ�ݵ�y�������۵���֤����y
h_lab_2=heart_scale_label(Size+1:2*Size);%�ڶ��ݵ�y, �Ƕ��۵���֤����y
h_lab_3= heart_scale_label(2*Size+1:3*Size);%�����ݵ�y����һ�۵���֤����y
h_inst_1=heart_scale_inst(1:Size,:); %�����۵���֤����x
h_inst_2=heart_scale_inst(Size+1: 2*Size,:); %�Ƕ��۵���֤����x
h_inst_3=heart_scale_inst(2*Size+1: 3*Size,:); %��һ�۵���֤����x
comb_lab_12=[h_lab_1; h_lab_2];%һ�۵�ѵ������y
comb_inst_12=[ h_inst_1; h_inst_2];% һ�۵�ѵ������x
comb_lab_13=[h_lab_1; h_lab_3];%���۵�ѵ������y
comb_inst_13=[ h_inst_1; h_inst_3];% ���۵�ѵ������x
comb_lab_23=[h_lab_2; h_lab_3];%���۵�ѵ������y
comb_inst_23=[ h_inst_2; h_inst_3]; %���۵�ѵ������x
rep_12= repmat(comb_lab_12,1,feature);
rep_13= repmat(comb_lab_13,1,feature);
rep_23= repmat(comb_lab_23,1,feature);
B_1=rep_12.*comb_inst_12;
B_2=rep_13.*comb_inst_13;
B_3=rep_23.* comb_inst_23;
Tf=3;%T_fold
Tfm1=Tf*m_1;
Tfm2=Tf*m_2;
m_hat=Tf*m_1+2*Tf*m_2;%C֮������ĸ���
B=[B_1,zeros(m_2, 2*feature);zeros(m_2, feature),B_2, zeros(m_2,feature);zeros(m_2,2*feature),B_3];
BB=B*B';
SP_1_Line2=[zeros(Tfm2,1),zeros(Tfm2,Tfm1),-BB,-eye(Tfm2)];
G_2=2*v(1)*v(Tfm1+Tfm2+2:m_hat+1)-v(Tfm1+2:Tfm1+Tfm2+1);
G=[-SP_1_Line2*v(1:m_hat+1)-ones(Tfm2,1);G_2];
c=[G.*(v(Tfm1+2:m_hat+1))-v(m_hat+2)*ones(2*Tfm2,1);-G_2];
ceq=[];
Q_1=[zeros(Tfm2,Tfm1+1),eye(Tfm2),zeros(Tfm2)];
dc=[diag(-SP_1_Line2*v(1:m_hat+1)-ones(Tfm2,1))*Q_1-diag(Q_1*v(1:m_hat+1))*SP_1_Line2; 2*v(Tfm1+Tfm2+2:m_hat+1).*v(Tfm1+Tfm2+2:m_hat+1),zeros(Tfm2,Tfm1),-diag(v(Tfm1+Tfm2+2:m_hat+1)),diag(4*v(1)*v(Tfm1+Tfm2+2:m_hat+1)-v(Tfm1+2:Tfm1+Tfm2+1));-2*v(Tfm1+Tfm2+2:m_hat+1),zeros(Tfm2,Tfm1),eye(Tfm2),-2*v(1)*eye(Tfm2)];
dc(1:2*Tfm2,m_hat+2)=0;
dceq=[];
end