include { GATK4_HAPLOTYPECALLER } from '../modules/nf-core/gatk4/haplotypecaller/main'
include { STAR_ALIGN            } from '../modules/nf-core/star/align/main'
include { SAMTOOLS_SORT         } from '../modules/nf-core/samtools/sort/main'
include { MULTIQC               } from '../modules/nf-core/multiqc/main'

workflow SAREK { }
