#!/bin/bash
#PBS -N uniprot
#PBS -e uniprot.err
#PBS -o uniprot.out
#PBS -l nodes=1:ppn=2
#PBS -l mem=8gb
#PBS -l walltime=4:00:00

work='/home/projects/ku_10024/people/zelili/berter'
data='/home/projects/ku_10024/data/databases/pubmed'

cd $work
if [ -f $work/uniprot_abstracts3.tsv ]; then
  rm -rf $work/uniprot_abstracts3.tsv
  touch $work/uniprot_abstracts3.tsv
else
  touch $work/uniprot_abstracts3.tsv
fi

#for i in $(cat $work/uniprot_pmid_uniq.txt); do
#  {
#    zgrep PMID:$i $data/pubmed*.tsv.gz | awk -F '\t' '{print $1"\t"$6}' | awk -F 'PMID:' '{print $2}' | uniq | sed 's/|.*\t/\t/g' >> $work/uniprot_abstracts2.tsv
#  }&
#done
for j in $(ls $data/pubmed*.tsv.gz | awk -F '/' '{print $NF}'); do
echo "#!/bin/bash
#PBS -N up${j}
#PBS -e up.err
#PBS -o up.out
#PBS -l nodes=1:ppn=20
#PBS -l mem=50gb
#PBS -l walltime=72:00:00

cd $work
for i in \$(cat $work/uniprot_pmid_uniq.txt); do
{
zgrep \"^PMID:${i}\\b\" $data/${j} | awk -F '\\t' '{print \$1\"\\t\"\$6}' | awk -F 'PMID:' '{print \$2}' | uniq | sed 's/|.*\\t/\\t/g' >> $work/uniprot_abstracts3.tsv
}&
done
" > $work/uniprot_abstract.sh
qsub < $work/uniprot_abstract.sh
done
