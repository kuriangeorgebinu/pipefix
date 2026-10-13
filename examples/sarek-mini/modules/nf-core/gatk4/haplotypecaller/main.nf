process GATK4_HAPLOTYPECALLER {
    tag "$meta.id"
    label 'process_medium'

    container 'broadinstitute/gatk:4.5.0.0'

    input:
    tuple val(meta), path(input), path(index)

    output:
    tuple val(meta), path("*.vcf.gz"), emit: vcf

    script:
    def avail_mem = task.memory ? (task.memory.mega*0.8).intValue() : 3072
    """
    gatk --java-options "-Xmx${avail_mem}M" HaplotypeCaller --input $input --output ${meta.id}.g.vcf.gz -ERC GVCF
    """
}
