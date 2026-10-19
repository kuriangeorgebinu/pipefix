process STAR_ALIGN {
    tag "$meta.id"
    label 'process_high'

    input:
    tuple val(meta), path(reads)

    output:
    tuple val(meta), path('*.bam'), emit: bam

    script:
    """
    STAR --runThreadN $task.cpus --genomeDir star --readFilesIn $reads
    """
}
