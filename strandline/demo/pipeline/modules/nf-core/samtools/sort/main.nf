process SAMTOOLS_SORT {
    tag "$meta.id"
    label 'process_medium'

    input:
    tuple val(meta), path(bam)

    output:
    tuple val(meta), path("*.bam"), emit: bam

    script:
    """
    samtools sort --threads $task.cpus -o ${meta.id}.sorted.bam $bam
    """
}
